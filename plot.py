from plotMaker import plot_data

# Single file example
#plot_data(
#    file1='fmo_cpp_state-6-initiation_6400-Pij.txt',
#    AU1='yes',
#    p1=[2,3]
#)

# Two file example
plot_data(
    file1='results.txt',
    AU1='yes',
    p1=[2,3],
#    file2='final_aggregated_py_fmo_run_1_py_time_dep_results.txt',
#    AU2='yes',
#    p2=[9,10,11,12,13,14,15]
)
