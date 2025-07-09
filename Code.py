#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import requests
import geopandas as gpd
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
import contextily as ctx

import numpy as np
import pandas as pd
from shapely.geometry import Point

################################################################################
# 1. Cities + categories + Population Data Path
################################################################################

# --- USER INPUT REQUIRED ---
POPULATION_DATA_FILE = "" 
# CSV has columns named 'X' (Longitude), 'Y' (Latitude), 'Z' (Population Density)
# --- END USER INPUT ---

cities = ["Nicosia, Cyprus", "Limassol, Cyprus", "Larnaca, Cyprus", "Paphos, Cyprus"]

service_categories = {
    "Healthcare": {
        "amenity": ["hospital", "clinic", "doctors", "pharmacy"]
    },
    "Education": {
        "amenity": ["school", "college", "university", "kindergarten"]
    },
    "FoodShops": {
        "shop": ["supermarket", "convenience", "greengrocer", "bakery"],
        "amenity": ["marketplace"]
    },
    "Leisure": {
        "leisure": ["park", "playground"],
        "amenity": ["community_centre", "library"]
    },
    "Culture": {
        "amenity": ["theatre", "cinema", "arts_centre", "museum"]
    },
    "Sports": {
        "leisure": ["sports_centre", "fitness_centre", "stadium", "gym"]
    },
    "Transport": {
        "amenity": ["bus_station"],
        "public_transport": ["stop_position", "platform"],
        "railway": ["station"]
    },
    "NeighborhoodSvc": {
        "amenity": ["bank", "post_office", "townhall", "cafe", "restaurant"]
    }
}

WALK_SPEED = 83.33      # ~5 km/h in m/min
GRID_SPACING = 500      # 500m
TIME_CUTOFF = 15        # 15 min
OVERPASS_URL = "https://overpass-api.de/api/interpreter"  


################################################################################
# 2. Overpass fallback with retry & bounding box (No changes needed)
################################################################################

def download_pois_overpass(boundary_polygon_latlon, tags_dict,
                           max_attempts=3, request_timeout=600, wait_between=10):
    """
    Download OSM POIs from Overpass within the bounding box of boundary_polygon_latlon (EPSG:4326).
    Then filter them to only keep those inside the boundary polygon.

    :param boundary_polygon_latlon: shapely Polygon in lat-lon (EPSG:4326)
    :param tags_dict: e.g. {"amenity": ["hospital", "clinic"], "shop":["supermarket"]}
    :param max_attempts: number of retries
    :param request_timeout: Overpass request timeout in seconds
    :param wait_between: seconds to wait between retries
    :return: GeoDataFrame of points inside polygon
    """
    # 1) bounding box in lat-lon
    minx, miny, maxx, maxy = boundary_polygon_latlon.bounds

    # 2) build Overpass QL for node-based queries
    or_clauses = []
    for key, vals in tags_dict.items():
        for v in vals:
            or_clauses.append(f'["{key}"="{v}"]')
    # chain them
    final_clause = ""
    for clause in or_clauses:
        final_clause += f"node{clause}({miny},{minx},{maxy},{maxx});"

    query = f"""
    [out:json][timeout:600];
    (
      {final_clause}
    );
    out center;
    """

    data_json = None
    for attempt in range(max_attempts):
        try:
            response = requests.post(OVERPASS_URL, data={"data": query}, timeout=request_timeout)
            response.raise_for_status()  # check for HTTP errors
            data_json = response.json()  # parse JSON
            break  # success
        except Exception as e:
            print(f"        Overpass request error (attempt {attempt+1}): {e}")
            if attempt < max_attempts - 1:
                print(f"        Retrying in {wait_between} seconds...")
                time.sleep(wait_between)
            else:
                print("        All attempts failed, returning empty GeoDataFrame.")
                return gpd.GeoDataFrame({'geometry': []}, crs="EPSG:4326")

    if not data_json or "elements" not in data_json:
        return gpd.GeoDataFrame({'geometry': []}, crs="EPSG:4326")

    # 3) parse Overpass JSON
    features = []
    for elem in data_json["elements"]:
        if elem["type"] == "node":
            # Ensure lat/lon are numbers
            try:
                lat = float(elem["lat"])
                lon = float(elem["lon"])
                geometry = Point(lon, lat)
                tags = elem.get("tags", {})
                features.append({"geometry": geometry, **tags})
            except (ValueError, TypeError):
                 print(f"        Warning: Skipping element due to invalid coordinates: {elem.get('id', 'N/A')}")


    gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
    if gdf.empty:
        return gdf

    # 4) filter strictly inside boundary polygon
    # Ensure boundary polygon is valid before using it
    if boundary_polygon_latlon.is_valid:
        gdf = gdf[gdf.geometry.within(boundary_polygon_latlon)] # Use within for strict containment
    else:
        print("        Warning: Boundary polygon is invalid. Cannot filter points inside.")
        return gpd.GeoDataFrame({'geometry': []}, crs="EPSG:4326") # Return empty if boundary invalid

    return gdf


################################################################################
# 3. Regular grid 
################################################################################

def create_grid_points(boundary_polygon, spacing=500, crs=None):
    minx, miny, maxx, maxy = boundary_polygon.bounds

    xs = np.arange(minx, maxx, spacing)
    ys = np.arange(miny, maxy, spacing)
    grid_points = []

    # Ensure boundary polygon is valid before using it
    if not boundary_polygon.is_valid:
        print("   Warning: Boundary polygon is invalid for grid creation. Returning empty grid.")
        return gpd.GeoDataFrame({'geometry': []}, crs=crs)

    for x in xs:
        for y in ys:
            p = Point(x, y)
            if boundary_polygon.contains(p):
                grid_points.append(p)

    gdf = gpd.GeoDataFrame({'geometry': grid_points}, crs=crs)
    return gdf


################################################################################
# 4. Main script
################################################################################

if __name__ == "__main__":

    city_results = []

    # Load population data ONCE before the loop
    try:
        pop_df = pd.read_csv(POPULATION_DATA_FILE)
        # Check for required columns
        if not {'X', 'Y', 'Z'}.issubset(pop_df.columns):
             raise ValueError("Population CSV must contain columns 'X', 'Y', and 'Z'")

        # Convert to GeoDataFrame
        pop_gdf_latlon = gpd.GeoDataFrame(
            pop_df,
            geometry=gpd.points_from_xy(pop_df.X, pop_df.Y),
            crs="EPSG:4326" # Assuming input is WGS84 Latitude/Longitude
        )
        print(f"Successfully loaded population data from {POPULATION_DATA_FILE}")
    except FileNotFoundError:
        print(f"Error: Population data file not found at {POPULATION_DATA_FILE}")
        print("Cannot proceed without population data. Exiting.")
        exit()
    except Exception as e:
        print(f"Error loading or processing population data: {e}")
        exit()


    for city in cities:
        print("="*70)
        print(f"Processing city: {city}")

        # 4A. Get city boundary from OSMnx geocoder
        try:
            city_boundary_gdf = ox.geocoder.geocode_to_gdf(city)
        except Exception as e:
            print(f"Could not fetch boundary for {city}: {e}")
            continue

        # keep largest polygon if multiple
        if len(city_boundary_gdf) > 1:
            orig_crs = city_boundary_gdf.crs # Store original CRS
            # Project to an equal-area projection suitable for area calculation if needed, or use projected later
            # For selecting largest, any planar projection works if areas are vastly different
            try:
                city_boundary_gdf_proj_select = city_boundary_gdf.to_crs(city_boundary_gdf.estimate_utm_crs())
                city_boundary_gdf = city_boundary_gdf.loc[[city_boundary_gdf_proj_select.geometry.area.idxmax()]]
            except Exception as e_proj:
                 print(f"   Warning: Could not project for area selection ({e_proj}). Selecting based on original CRS area.")
                 # Fallback if projection fails, might be less accurate for area comparison
                 city_boundary_gdf = city_boundary_gdf.loc[[city_boundary_gdf.geometry.area.idxmax()]]


        boundary_polygon = city_boundary_gdf.geometry.iloc[0]

        # save boundary
        boundary_path = f"{city.split(',')[0].replace(' ', '_').lower()}_boundary.geojson"
        city_boundary_gdf.to_file(boundary_path, driver="GeoJSON")
        boundary_latlon_poly = city_boundary_gdf.to_crs("EPSG:4326").geometry.iloc[0] # For POI download

        # 4B. Download street network
        print("    Downloading street network...")
        try:
            G = ox.graph_from_polygon(boundary_polygon, network_type="walk", simplify=False)
        except Exception as e:
             print(f"    Error downloading network for {city}: {e}. Skipping city.")
             continue # Skip to next city if network download fails


        
        try:
            if nx.is_directed(G):
                 component_nodes = max(nx.strongly_connected_components(G), key=len)
            else:
                 component_nodes = max(nx.weakly_connected_components(G), key=len)
            G = G.subgraph(component_nodes).copy() # Use copy to avoid modifying original during iteration if needed elsewhere
        except Exception as e_comp:
             print(f"    Warning: Could not isolate largest component ({e_comp}). Using potentially disconnected graph.")
             # Fallback: remove only completely isolated nodes
             iso_nodes = list(nx.isolates(G))
             G.remove_nodes_from(iso_nodes)


        # simplify
        G_simpl = ox.simplify_graph(G)

        # project the graph
        try:
            G_proj = ox.project_graph(G_simpl)
        except Exception as e_proj_graph:
            print(f"    Error projecting graph for {city}: {e_proj_graph}. Skipping city.")
            continue # Skip if projection fails

        target_crs = G_proj.graph["crs"]
        boundary_gdf_proj = city_boundary_gdf.to_crs(target_crs)
        boundary_polygon_proj = boundary_gdf_proj.geometry.iloc[0]

        area_m2 = boundary_polygon_proj.area
        area_km2 = area_m2/1e6 if area_m2>0 else 0

        

        # 4D. Road density
        print("    Calculating road density...")
        edges_len_m = [d["length"] for _,_,d in G_proj.edges(data=True) if "length" in d]
        total_road_km = sum(edges_len_m)/1000.0
        road_density = total_road_km/area_km2 if area_km2>0 else 0

        
        print("    Downloading transit stops...")
        transit_tags = {
            "highway":["bus_stop"],
            "amenity":["bus_station"],
            "railway":["station","halt"]
        }
        # Use the pre-calculated latlon boundary polygon
        transit_gdf = download_pois_overpass(boundary_latlon_poly, transit_tags)
        transit_stop_density = len(transit_gdf)/area_km2 if area_km2>0 else 0

        # 4F. 15-min city analysis
        print("    Performing 15-minute city analysis...")
        # 4F-i) create a sampling grid
        grid_gdf_proj = create_grid_points(boundary_polygon_proj, spacing=GRID_SPACING, crs=target_crs)

        if grid_gdf_proj.empty:
             print("    Analysis grid is empty (possibly due to small or invalid boundary). Skipping 15-min analysis.")
             # Still append results with 0 scores if desired, or skip city entirely
             avg_score = 0
             pct_full = 0
             avg_pop_density = 0
             pop_wtd_avg_score = 0
             # Go straight to appending results
        else:
            # --- Integrate Population Data ---
            print("      Integrating population density...")
            try:
                # Project population data to the analysis CRS
                pop_gdf_proj = pop_gdf_latlon.to_crs(target_crs)

                # Spatially join grid points to nearest population point
                # `sjoin_nearest` finds the nearest feature in 'right_gdf' (pop_gdf_proj) for each feature in 'left_gdf' (grid_gdf_proj)
                grid_gdf_proj = gpd.sjoin_nearest(grid_gdf_proj, pop_gdf_proj[['geometry', 'Z']], how='left', max_distance=GRID_SPACING*2) # Added max_distance limit
                # Rename the joined population column ('Z')
                grid_gdf_proj.rename(columns={'Z': 'population_density'}, inplace=True)
                # Fill points with no nearby population data with 0 density
                grid_gdf_proj['population_density'].fillna(0, inplace=True)
                # Ensure density is not negative
                grid_gdf_proj['population_density'] = grid_gdf_proj['population_density'].clip(lower=0)

                # Clean up index names if they conflict (common after sjoin)
                grid_gdf_proj = grid_gdf_proj.rename(columns=lambda x: x.replace('index_', 'idx_'))

            except Exception as e_pop_join:
                 print(f"      Warning: Could not join population data to grid ({e_pop_join}). Proceeding without population weighting.")
                 grid_gdf_proj['population_density'] = 1 # Assign weight 1 if join fails, so weighted avg = simple avg


            # 4F-ii) Download amenities
            amenity_points_cat = {}
            for cat_name, tag_dict in service_categories.items():
                print(f"      Downloading amenities for category: {cat_name}")
                # Use the pre-calculated latlon boundary polygon
                cat_pois = download_pois_overpass(boundary_latlon_poly, tag_dict, max_attempts=3, request_timeout=600)
                if not cat_pois.empty:
                    cat_pois = cat_pois.to_crs(target_crs)
                amenity_points_cat[cat_name] = cat_pois

            # 4F-iii) add travel_time edge attribute
            for u,v,data in G_proj.edges(data=True):
                ln = data.get("length", None)
                if ln and ln > 0: # Ensure length is positive
                    data["travel_time"] = ln / WALK_SPEED
                else:
                    data["travel_time"] = float('inf') # Use infinity for unreachable edges

            # 4F-iv) find nearest node for each amenity
            print("      Finding nearest network nodes for amenities...")
            for cat_name, cat_gdf in amenity_points_cat.items():
                if cat_gdf.empty or 'geometry' not in cat_gdf.columns:
                     cat_gdf["nearest_node"] = np.nan
                     continue

                cat_gdf_points = cat_gdf[cat_gdf.geometry.geom_type == 'Point']
                if not cat_gdf_points.empty:
                    xs = cat_gdf_points.geometry.x
                    ys = cat_gdf_points.geometry.y
                    # Use lists for nearest_nodes function
                    nearest_nodes = ox.distance.nearest_nodes(G_proj, xs.to_list(), ys.to_list())
                    cat_gdf.loc[cat_gdf_points.index, "nearest_node"] = nearest_nodes
                else:
                    # Handle cases where amenities might be lines/polygons without a clear point geometry
                     cat_gdf["nearest_node"] = np.nan


            # 4F-v) evaluate 15-min from each grid cell
            print("      Evaluating accessibility from grid points...")
            total_cat = len(service_categories)
            scores = []

            if not grid_gdf_proj.empty:
                # Pre-calculate nearest nodes for grid points
                grid_nodes_start = ox.distance.nearest_nodes(G_proj, grid_gdf_proj.geometry.x.to_list(), grid_gdf_proj.geometry.y.to_list())
                grid_gdf_proj['nearest_node_grid'] = grid_nodes_start # Use a distinct name
            else:
                 grid_nodes_start = []


            # Pre-compute unique destination nodes per category (ensure they are in the graph)
            category_nodes = {}
            graph_nodes_set = set(G_proj.nodes()) # Get all nodes in the projected graph
            for cat_name, cat_gdf in amenity_points_cat.items():
                 valid_nodes = set()
                 if not cat_gdf.empty and 'nearest_node' in cat_gdf.columns:
                     # Drop NaN, ensure nodes are valid graph nodes
                     potential_nodes = cat_gdf['nearest_node'].dropna().unique()
                     valid_nodes = {node for node in potential_nodes if node in graph_nodes_set}
                 category_nodes[cat_name] = valid_nodes


            # Iterate through grid points
            for idx, cell in grid_gdf_proj.iterrows():
                node_start = cell['nearest_node_grid']

                if pd.isna(node_start) or node_start not in graph_nodes_set:
                     scores.append(0) # Assign score 0 if start node is invalid or not in graph
                     continue

                try:
                    dist_dict = nx.single_source_dijkstra_path_length(
                        G_proj, node_start, cutoff=TIME_CUTOFF, weight="travel_time"
                    )
                    reachable_nodes = set(dist_dict.keys())

                except Exception as e:
                    # print(f"      Dijkstra error for grid cell {idx} starting at node {node_start}: {e}") # Verbose
                    reachable_nodes = set() # Treat as no nodes reachable on error

                reachable_categories_count = 0
                for cat_name, target_nodes_set in category_nodes.items():
                    if not target_nodes_set.isdisjoint(reachable_nodes):
                        reachable_categories_count += 1

                score = (reachable_categories_count / total_cat) * 100 if total_cat > 0 else 0
                scores.append(score)

            grid_gdf_proj["score_15min"] = scores

            # --- Calculate Final Metrics ---
            # Average 15 minute Score
            avg_score = grid_gdf_proj["score_15min"].mean() if not grid_gdf_proj.empty else 0
            # Percentage of points with full access
            pct_full = (grid_gdf_proj["score_15min"] == 100).mean() * 100 if not grid_gdf_proj.empty else 0
            # Average Population Density (of grid points where pop data was found)
            avg_pop_density = grid_gdf_proj['population_density'][grid_gdf_proj['population_density'] > 0].mean() if 'population_density' in grid_gdf_proj.columns and (grid_gdf_proj['population_density'] > 0).any() else 0

            # Population Weighted Score
            if 'population_density' in grid_gdf_proj.columns and grid_gdf_proj['population_density'].sum() > 0:
                pop_wtd_avg_score = np.average(
                    grid_gdf_proj['score_15min'],
                    weights=grid_gdf_proj['population_density']
                )
            else:
                 # If no population data or sum is zero, weighted average is same as simple average
                 pop_wtd_avg_score = avg_score


            # --- Save outputs ---
            city_clean = city.split(",")[0].replace(" ","_").lower()
            shp_file = f"{city_clean}_15min_score_pop.shp" # Modified name
            csv_file = f"{city_clean}_15min_score_pop.csv" # Modified name

            if not grid_gdf_proj.empty:
                try:
                    # Select columns to save - include population density
                    cols_to_save = ['geometry', 'score_15min', 'population_density']
                    save_gdf = grid_gdf_proj[[col for col in cols_to_save if col in grid_gdf_proj.columns]]
                    save_gdf.to_file(shp_file, driver="ESRI Shapefile")

                    grid_wgs = save_gdf.to_crs("EPSG:4326")
                    grid_wgs["lat"] = grid_wgs.geometry.y
                    grid_wgs["lon"] = grid_wgs.geometry.x
                    # Ensure score and pop density are included
                    grid_wgs_csv = grid_wgs[["lat", "lon", "score_15min", "population_density"]].copy()
                    grid_wgs_csv.to_csv(csv_file, index=False)
                except Exception as e:
                     print(f"      Error saving output files for {city_clean}: {e}")
            else:
                print(f"      Grid for {city_clean} is empty. Skipping file saving.")


        # --- Append results for the city ---
        city_results.append({
            "City": city,
            "Area_km2": round(area_km2, 2),
            "WalkableRoad_km_per_km2": round(road_density, 2),
            "TransitStop_km2": round(transit_stop_density, 2),
            "AvgPopDensity_grid": round(avg_pop_density, 2), # Avg density over grid points
            "Avg15minScore": round(avg_score, 3), # Higher precision
            "PctAllServices15min": round(pct_full, 3), # Higher precision
            "PopWtdAvg15minScore": round(pop_wtd_avg_score, 3) # New weighted score
        })

        print(f"    Done with {city}. 15-min avg={avg_score:.3f}, full={pct_full:.3f}%, PopWtdAvg={pop_wtd_avg_score:.3f}")


    # --- Final Summary ---
    if city_results:
        df = pd.DataFrame(city_results)
        output_csv_filename = "cyprus_accessibility_pop_15min_index.csv" # Modified name
        df.to_csv(output_csv_filename, index=False)
        print("\n" + "="*70)
        print("Summary of results (Accessibility + Population):")
        print(df.to_string()) # Use to_string to prevent wrapping
        print(f"\nResults saved to {output_csv_filename}")
        print("="*70)
    else:
        print("No results to summarize.")




##############################################################################
# 6. Visualization of 15-minute city scores
##############################################################################

def plot_15min_map(shapefile_path, city_name, output_png="map.png", score_col='score_15min'):
    """
    Reads a shapefile of 15-min city scores, plots it with a basemap, saves as PNG.
    Allows specifying the column to plot.
    """
    try:
        gdf = gpd.read_file(shapefile_path)

        if gdf.empty or score_col not in gdf.columns:
             print(f"      Warning: Shapefile {shapefile_path} is empty or missing '{score_col}'. Skipping plot.")
             return

        gdf = gdf.to_crs(epsg=3857) # Project to Web Mercator

        fig, ax = plt.subplots(figsize=(10, 8))
        gdf.plot(column=score_col, cmap='viridis', legend=True, alpha=0.7, ax=ax,
                 legend_kwds={'label': f"{score_col} (%)", 'orientation': 'vertical'})

        try:
             ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)
        except Exception as e:
             print(f"      Warning: Could not add basemap for {city_name}. Plotting without basemap. Error: {e}")

        ax.set_title(f"15-Minute Accessibility ({score_col}): {city_name}", fontsize=16)
        ax.axis('off')

        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        print(f"      Map saved to {output_png}")
        plt.close(fig) # Close the figure

    except Exception as e:
         print(f"      Error plotting map for {city_name} from {shapefile_path}: {e}")

# Automatically plot maps for each city processed above
print("\n" + "="*70)
print("Generating Maps:")
print("="*70)
for city in cities:
    city_clean = city.split(",")[0].replace(" ", "_").lower()
    # Use the updated shapefile name
    shp_file = f"{city_clean}_15min_score_pop.shp"
    # Create separate maps for score and density
    output_png_score = f"{city_clean}_15min_map_score.png"
    output_png_pop = f"{city_clean}_15min_map_pop_density.png"

    import os
    if os.path.exists(shp_file):
        print(f"  Plotting maps for {city}...")
        # Plot score
        plot_15min_map(shp_file, city, output_png_score, score_col='score_15min')
        # Plot population density (check if column exists)
        temp_gdf = gpd.read_file(shp_file) # Read quickly to check cols
        if 'population_density' in temp_gdf.columns:
             # Modify the function slightly or just replot with different column/label for density
             try:
                  gdf = temp_gdf.to_crs(epsg=3857)
                  fig, ax = plt.subplots(figsize=(10, 8))
                  # Use a different colormap perhaps for density, like 'magma' or 'plasma'
                  gdf.plot(column='population_density', cmap='magma', legend=True, alpha=0.7, ax=ax,
                           legend_kwds={'label': "Population Density (Z value)", 'orientation': 'vertical'})
                  try:
                       ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)
                  except Exception as e: print(f"Warning: No basemap for density plot. {e}")
                  ax.set_title(f"Population Density Grid: {city_name}", fontsize=16)
                  ax.axis('off')
                  plt.savefig(output_png_pop, dpi=300, bbox_inches='tight')
                  print(f"      Map saved to {output_png_pop}")
                  plt.close(fig)
             except Exception as e_plot_pop:
                  print(f"      Error plotting population density map: {e_plot_pop}")
        else:
             print(f"      'population_density' column not found in {shp_file}. Skipping density map plot.")


    else:
        print(f"  Shapefile not found for {city} ({shp_file}). Skipping map plots.")

print("\nScript finished.")