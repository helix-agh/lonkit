# LON Module

::: lonkit.lon.LON
    options:
      show_root_heading: true
      show_source: true
      members:
        - from_trace_data
        - n_vertices
        - n_edges
        - vertex_names
        - vertex_fitness
        - vertex_count
        - get_sinks
        - compute_metrics
        - to_cmlon

::: lonkit.lon.CMLON
    options:
      show_root_heading: true
      show_source: true
      members:
        - from_lon
        - n_vertices
        - n_edges
        - vertex_fitness
        - vertex_count
        - get_sinks
        - get_global_sinks
        - get_local_sinks
        - compute_metrics

::: lonkit.lon.LONConfig
    options:
      show_root_heading: true
      show_source: true
      members:
        - fitness_aggregation
        - warn_on_duplicates
        - max_fitness_deviation
        - eq_atol
