import math
import numpy as np
import pandas as pd
import tqdm
from scipy.sparse import coo_matrix
from typing import List, Optional

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.collections import PathCollection
import matplotlib.transforms as mtransforms
from matplotlib.lines import lineMarkers

from utils import Multidict, get_angle, get_lower_bounds
from database import Graph, CelestialGraph
#from solver import AngularGraphSolution

def visualize_graph_2d(graph: Graph, savePath=None):
    fig = plt.figure()
    axis = plt.subplot()
    axis.axis('off')
    
    _visualize_edges_2d(graph)
    _visualize_vertices_2d(graph)
    _visualize_celest_body_2d(axis, graph)
    if savePath:
        
        if savePath[-3:] == "ipe":
            old_backend = matplotlib.get_backend()
            matplotlib.use('module://backend_ipe')
            save_format = "ipe"
            plt.savefig(savePath, format=save_format)
            matplotlib.use(old_backend)
        else:            
            plt.savefig(savePath)
        
        
    else:    
        plt.show()

def visualize_min_sum_sol_2d(solution: 'AngularGraphSolution'):
    graph = solution.graph
    fig = plt.figure()
    axis = plt.subplot()
    axis.axis('off')
    
    _visualize_edges_2d(solution.graph)
    _visualize_vertices_2d(solution.graph)
    _visualize_celest_body_2d(axis, solution.graph)

    # Make an edge order for vertices
    vertex_order = Multidict()
    ordered_times = solution.get_ordered_times()
    for time_key in ordered_times.get_ordered_keys():
        for edges in ordered_times[time_key]:
            if edges[0] < edges[1]:
                vertex_order[edges[0]] = edges
                vertex_order[edges[1]] = edges
    # Get minimum edge length
    min_length = max(np.array(
        [
            np.linalg.norm(solution.graph.vertices[i] - solution.graph.vertices[j])
            for i, j in solution.graph.edges
        ]
    ).min(), 0.4)
    # Draws the angle paths in a circular fashion
    path_list = []
    last_points = []
    for vertex_key in vertex_order:
        last_edge = None
        last_direction = None
        current_min_length = min_length * 0.3
        last_point = None
        for edge in vertex_order[vertex_key]:
            if last_edge:
                other_vertices = np.hstack([
                    np.setdiff1d(np.array(last_edge), np.array([vertex_key])),
                    np.setdiff1d(np.array(edge), np.array([vertex_key]))
                    ])
                

                angles = [get_angle(
                    graph.vertices[vertex_key],
                    graph.vertices[vertex_key] + [1, 0],
                    graph.vertices[other_vertex]) for other_vertex in other_vertices]

                # If y-coord is below the current vertex we need to calculate the angle different
                for i in range(len(angles)):
                    if graph.vertices[other_vertices[i]][1] < graph.vertices[vertex_key][1]:
                        angles[i] = 360 - angles[i]

                # Calculate if we need to go from angle[0] to angle[1] or other way around
                # to not create an arc over 180 degrees
                diff = abs(angles[0] - angles[1])
                if diff > 180:
                    diff = 360 - diff
                normal_angle_direction = math.isclose((angles[0] + diff) % 360, angles[1], rel_tol=1e-5)
                if not normal_angle_direction:
                    angles = reversed(angles)

                # 1 shall be clockwise and -1 counter-clockwise direction
                current_direction = 1 if normal_angle_direction else -1

                if last_direction:
                    if current_direction != last_direction: # direction change happened
                        current_min_length *= 1.25

                # Transform the arc to the right position
                transform = mtransforms.Affine2D().scale(current_min_length, current_min_length)
                transform = transform.translate(*graph.vertices[vertex_key])
                arc = Path.arc(*angles)
                arc_t = arc.transformed(transform)
                
                
                if last_direction:
                    if current_direction != last_direction: # direction change happened
                        last_vertex = path_list[-1].vertices[-1] if last_direction == 1 else path_list[-1].vertices[0]
                        new_vertex = arc_t.vertices[0] if current_direction == 1 else arc_t.vertices[-1]
                        bridge_path = Path([last_vertex, new_vertex])
                        path_list.append(bridge_path)

                last_direction = current_direction
                path_list.append(arc_t)
                last_point = path_list[-1].vertices[-1] if last_direction == 1 else path_list[-1].vertices[0]
                last_points.append(last_point)
            last_edge = edge
        # Add these points to detect direction
        last_points.append(last_point)

    path_collection = PathCollection(path_list, edgecolor='r', facecolor='#00000000')
    axis.add_collection(path_collection)
    a_last_points = np.array([l for l in last_points if l is not None])
    plt.plot(a_last_points[:, 0], a_last_points[:, 1], 'r.')
    axis.autoscale()
    plt.show()

def visualize_solution_2d(solution: 'AngularGraphSolution', title=None, show_used=True):
    fig = plt.figure()
    if title:
        fig.suptitle(title)
    #fig.subplots_adjust(hspace=0.3, wspace=0.3)
    
    ordered_times = solution.get_ordered_times()
    cells_needed = len(ordered_times)
    row_num, col_num = _calculate_row_col_needed(cells_needed)
    fig.set_size_inches(fig.get_size_inches()[1], fig.get_size_inches()[1])
    i = 1
    already_used = []
    for time in ordered_times.get_ordered_keys():
        axis = plt.subplot(row_num, col_num, i)
        #if solution.solution_type in ["makespan"]:
        plt.title("t = {0}".format(round(time, 2)))
        axis.axis('off')
        
        _visualize_edges_2d(
            solution.graph,
            ordered_times[time], already_used)
        _visualize_vertices_2d(solution.graph)

        if show_used:
            already_used.extend(ordered_times[time])
        _visualize_celest_body_2d(axis, solution.graph)
        i += 1
    fig.tight_layout()
    plt.show()

def _visualize_edges_2d(graph: Graph, taken_edges=None, already_used=None):
    if graph.vertices.dtype == np.dtype('O'):
        graph.vertices = np.array([p for p in graph.vertices])
    for edge in graph.edges:
        plt.plot(graph.vertices[edge][:, 0], graph.vertices[edge][:, 1], color='black', marker=',', alpha=0.3)
    if already_used:
        for indices in already_used:
            edge = np.array([graph.vertices[i] for i in indices])
            plt.plot(edge[:, 0], edge[:, 1], "y-")
    if taken_edges:
        for indices in taken_edges:
            edge = np.array([graph.vertices[i] for i in indices])
            plt.plot(edge[:, 0], edge[:, 1], "r-")

def _visualize_vertices_2d(graph: Graph):
    plt.plot(graph.vertices[:, 0], graph.vertices[:, 1], "b.")

def _visualize_celest_body_2d(axis, graph: Graph):
    if isinstance(graph, CelestialGraph):
        for body in graph.celestial_bodies:
            # Add earth as celestial object
            image = plt.imread("utils/figures/world-1303628_1920.png")
            radius = 870
            scale = len(image) / (radius*2)
            extent = (
                (body.position[0] - float(body.size)) * scale,
                (body.position[0] + float(body.size)) * scale,
                (body.position[1] - float(body.size)) * scale,
                (body.position[1] + float(body.size)) * scale
            )

            im = axis.imshow(image, extent=extent)
            pos = body.position
            patch = patches.Circle(pos, radius=float(body.size), transform=axis.transData)
            im.set_clip_path(patch)
            axis.autoscale_view()
            
def _visualize_celest_body_2d_old(axis, graph: Graph):
    if isinstance(graph, CelestialGraph):
        for body in graph.celestial_bodies:
            # Add earth as celestial object
            image = plt.imread("utils/figures/720px-The_Earth_seen_from_Apollo_17.jpg")
            radius = 320
            scale = len(image) / (radius*2)
            extent = (
                (body.position[0] - float(body.size)) * scale,
                (body.position[0] + float(body.size)) * scale,
                (body.position[1] - float(body.size)) * scale,
                (body.position[1] + float(body.size)) * scale
            )

            im = axis.imshow(image, extent=extent)
            pos = body.position
            patch = patches.Circle(pos, radius=float(body.size), transform=axis.transData)
            im.set_clip_path(patch)
            axis.autoscale_view()
            

def _calculate_row_col_needed(cells_needed: int):
     # Calculate the quadratic amount needed
     # Aim is to get it as quadratic as possible
     # Maybe later aim to get a ratio near display ratio?
    quad_num = math.ceil(math.sqrt(cells_needed))
    # Calculate how many rows are now actually needed
    row_num = math.ceil(cells_needed / quad_num)
    return row_num, quad_num

_sol_type_to_label = {"runtime": "Runtime", "min_sum": "MinSum", "local_min_sum": "LocalMinSum", "makespan": "Makespan"}

class VisTypes:
        Absolute = 0
        VsBest = 1
        VsLB = 2
        All = 3
        LB_Runtime = 4

# From https://stackoverflow.com/questions/55767312/how-to-position-suptitle
def _make_space_above(axes, topmargin=1): 
    """ increase figure size to make topmargin (in inches) space for 
        titles, without changing the axes sizes"""
    fig = axes.flatten()[0].figure
    s = fig.subplotpars
    w, h = fig.get_size_inches()

    figh = h - (1-s.top)*h  + topmargin
    fig.subplots_adjust(bottom=s.bottom*h/figh, top=1-topmargin/figh)
    fig.set_figheight(figh)

def visualize_solution_scatter(jobs: List['TaskJobs'], title,
                               path: Optional[str]=None, solution_type: Optional[str]=None,
                               logscale=False, ylabel=None, vis_type=VisTypes.Absolute,
                               loc=1, bbox_pos=None, top_margin=0.65,
                               show_legend=True):
    if solution_type is None:
        solution_type = _get_dominant_solution_type(jobs)
    if not ylabel:
        y_label = _sol_type_to_label[solution_type]
    for s in tqdm.tqdm(jobs, desc="Calculate lower bounds"):
        if s.solution is not None:
            _get_LB(s.solution, solution_type)
    df = pd.DataFrame(
        [
            {
            "Solver": _get_solver_name(job),
            "VertAmount": job.solution.graph.vert_amount,
            "EdgeAmount": job.solution.graph.edge_amount,
            "Graph_id": job.solution.graph.id,
            "Runtime": float(job.solution.runtime) if job.prev_job is None else float(job.prev_job.solution.runtime+job.solution.runtime),
            "MinSum": job.solution.min_sum,
            "LocalMinSum": job.solution.local_min_sum,
            "Makespan": job.solution.makespan,
            "LB": _get_LB(job.solution, solution_type)}
            for job in tqdm.tqdm(jobs, desc="Collect solution information") if job.solution is not None
            ])
    # Then plot the data
    if vis_type == VisTypes.All:
        fig, axes = plt.subplots(nrows=2,ncols=2, sharex=True)
        #fig.suptitle(title)
        if isinstance(logscale, bool):
            logscale = [logscale for i in range(4)]
        if len(logscale) < 4:
            logscale = logscale + [False for i in range(4-len(logscale))]
        label_cols = 3
        top_margin = top_margin+0.2
        columns = _plot_data(df, solution_type, "Edge amount", y_label, logscale=logscale[0], vis_type=VisTypes.Absolute, ax=axes[0,0],)# show_legend=True)
        columns = _plot_data(df, solution_type, "Edge amount", y_label, logscale=logscale[1], vis_type=VisTypes.VsBest, ax=axes[0,1],)# show_legend=True)
        columns = _plot_data(df, solution_type, "Edge amount", y_label, logscale=logscale[2], vis_type=VisTypes.VsLB, ax=axes[1,0],)# show_legend=True)
        columns = _plot_data(df, "runtime", "Edge amount", "Runtime", logscale=logscale[3], vis_type=VisTypes.Absolute, ax=axes[1,1],)# show_legend=True)
        fig.set_size_inches(fig.get_size_inches()*1.5)
        fig.tight_layout()
        handles, labels = axes[1, 1].get_legend_handles_labels()
        fig.legend(handles, labels, loc=loc, bbox_to_anchor=bbox_pos,\
            ncol=label_cols)
        
        _make_space_above(axes, top_margin)
        #fig.legend([m_i[1] for m_i in columns], loc=loc, bbox_to_anchor=bbox_pos)
    elif vis_type == VisTypes.LB_Runtime:
        fig, axes = plt.subplots(nrows=1,ncols=2, sharex=True)
        #fig.suptitle(title)
        if isinstance(logscale, bool):
            logscale = [logscale for i in range(2)]
        if len(logscale) < 4:
            logscale = logscale + [False for i in range(2-len(logscale))]
        label_cols = 3
        top_margin = top_margin+0.25
        columns = _plot_data(df, solution_type, "Edge amount", y_label, logscale=logscale[0], vis_type=VisTypes.VsLB, ax=axes[0],)# show_legend=True)
        columns = _plot_data(df, "runtime", "Edge amount", "Runtime", logscale=logscale[1], vis_type=VisTypes.Absolute, ax=axes[1],)# show_legend=True)
        fig.set_size_inches(fig.get_size_inches()*(1.3, 0.9))
        fig.tight_layout()
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc=loc, bbox_to_anchor=bbox_pos,\
            ncol=label_cols)
        _make_space_above(axes, top_margin)
        
    else:
        columns = _plot_data(df, solution_type, "Edge amount", y_label, logscale=logscale, vis_type=vis_type, show_legend=True)
        plt.title(title)

    
    if path is None:
        plt.show()
    else:
        if path[-3:] == "ipe":
            old_backend = matplotlib.get_backend()
            matplotlib.use('module://backend_ipe')
            save_format = "ipe"
            plt.savefig(path, format=save_format)
            matplotlib.use(old_backend)
        else:
            plt.savefig(path)

def _get_LB(sol: "AngularGraphSolution", solution_type):
    graph = sol.graph
    if solution_type == "local_min_sum":
        try:
            return _get_LB.local_min_sum_lbs[graph.id]
        except KeyError:
            lb = max(get_lower_bounds(graph))
            _get_LB.local_min_sum_lbs[graph.id] = lb
            return lb
    if solution_type == "min_sum":
        try:
            return _get_LB.min_sum_lbs[graph.id]
        except KeyError:
            lb = sum(get_lower_bounds(graph))
            _get_LB.min_sum_lbs[graph.id] = lb
            return lb
    if solution_type == "makespan":
        try:
            lb = _get_LB.makespan_lbs[graph.id]
        except KeyError:
            from solver.coloring_solver import Coloring_CP_Solver
            from pyclustering.gcolor.dsatur import dsatur
            if graph.edge_amount < 40:
                solver = Coloring_CP_Solver()
                colors = solver.solve(graph)
            else:
                dsatur_instance = dsatur(graph.ad_matrix)
                dsatur_instance.process()
                colors = dsatur_instance.get_colors()

            lb = ((math.ceil(math.log2(max(colors)))-2) / 2) * 90
            _get_LB.makespan_lbs[graph.id] = lb
        if sol.makespan and lb > sol.makespan:
            
            log_c_number = math.ceil(sol.makespan * 2 / 90) + 2
            lb2 = ((math.ceil(log_c_number)-2) / 2) * 90
            if lb > lb2:
                _get_LB.makespan_lbs[graph.id] = lb2
                lb = lb2
        return lb
_get_LB.min_sum_lbs = {}
_get_LB.local_min_sum_lbs = {}
_get_LB.makespan_lbs = {}

def _get_dominant_solution_type(jobs: List['TaskJobs']):
    sol_type = np.array([job.solution.solution_type for job in tqdm.tqdm(jobs, desc="Load solutions") if job.solution is not None])
    types, counter = np.unique(sol_type, return_counts=True)
    max_index = np.argmax(counter)
    return types[max_index]

MARKERS = ['o', '^', 'v', 'h', '*', 'x', 'd', 'P', '1', '.']



def _plot_data(data: pd.DataFrame, solution_type, xlabel, ylabel, logscale=False, markers: Optional[List[str]] = None, vis_type=0, ax: Optional[plt.Axes] = None,
               show_legend=False):
    if not markers:
        markers = MARKERS
    markers = [markers[i % len(markers)] for i in range(len(data))]
    label = _sol_type_to_label[solution_type]
    reshaped = data.pivot_table(data, index=['Graph_id', 'EdgeAmount'], columns=['Solver'])

    if vis_type == VisTypes.VsBest:
        mins = data[['Graph_id', 'Solver', 'EdgeAmount', label]]\
        .groupby(['Graph_id']).min()
        mins = mins.pivot_table(mins, index=["Graph_id", "EdgeAmount"])
        plot_table = reshaped[label].divide(mins[label], axis=0).reset_index()
    elif vis_type == VisTypes.VsLB:
        l_bs = data[['Graph_id', 'EdgeAmount', "LB"]]
        l_bs = l_bs.pivot_table(l_bs, index=["Graph_id", "EdgeAmount"])
        plot_table = reshaped[label].divide(l_bs["LB"], axis=0).reset_index()
        plot_table = plot_table.replace([np.inf, -np.inf], np.nan)
    else:
        plot_table = reshaped[label].reset_index()
    if ax:
        plot_table.plot(1, range(2, len(plot_table.columns)), ax=ax, alpha=0.5)
    else:
        ax = plot_table.plot(1, range(2, len(plot_table.columns)))
    for i, line in enumerate(ax.get_lines()):
        line.set_marker(MARKERS[i])
        line.set_linestyle('')
    if not show_legend:
        ax.legend().set_visible(False)
    ax.xaxis.label.set_text(xlabel)

    
    if logscale:
        ax.set_yscale("log")
        #l = _calculate_labels(ax, plot_table)
        #ax.set_yticks(l)
        #ax.set_yticklabels(l)
    if vis_type == VisTypes.VsBest:
        ax.yaxis.label.set_text(f"{ylabel} / best sol")
    elif vis_type == VisTypes.VsLB:
        ax.yaxis.label.set_text(f"{ylabel} / lower bound")
        if solution_type == "makespan" and data["EdgeAmount"].max() > 40:
            ax.yaxis.label.set_text(f"{ylabel} / heuristical bound")
            """if max(plot_table[solver_columns].max()) < 25:
                ax.set_yticks([1,5,10,20])
                ax.set_yticklabels([1,5,10,20])
            elif max(plot_table[solver_columns].max()) < 100:
                l = [1,10,20,50,80]
                ax.set_yticks(l)
                ax.set_yticklabels(l)
            elif max(plot_table[solver_columns].max()) < 3.25:
                ax.set_yticks(np.arange(1, 3.25, step=0.25))
                ax.set_yticklabels(np.arange(1, 3.25, step=0.25))"""
    else:        
        ax.yaxis.label.set_text(ylabel)
    
    
    return reshaped.columns

def _get_solver_name(job: "TaskJobs"):
    name = job.solution.solver if job.prev_job is None else job.prev_job.solution.solver+"+"+job.solution.solver
    name = name.replace("Angular","")
    return name

def _calculate_labels(ax, plot_table):
    solver_columns = [solver_columns for solver_columns in plot_table.columns if solver_columns not in ["Graph_id", "EdgeAmount"]]
    min_l = min(plot_table[solver_columns].min())
    max_l = max(plot_table[solver_columns].max())
    min_10_base = math.floor(math.log10(min_l))
    max_10_base = math.ceil(math.log10(max_l))
    if min_10_base + 1 == max_10_base and (max_l - min_l < 3.25 * 10**min_10_base):
        if max_l - min_l < 3.25 * 10**min_10_base:
            l = np.arange(1*10**min_10_base, max_l, step=0.2*10**min_10_base)
        if max_l - min_l > 3.25 * 10**min_10_base:
            l = np.arange(1*10**min_10_base, max_l, step=0.25*10**min_10_base)
    else:
        l_base = [i for i in range(1,10)]
        if min_10_base +2  <= max_10_base:
            l_base = [1,2,5]
        if min_10_base + 4 <= max_10_base:
            l_base = [1, 5]
        if min_10_base + 8 < max_10_base:
            l_base = [1]
        l = []
        multiplicator = 10**-3
        while not l or (l[-1] < max_l*1.2 and multiplicator < max_l*1.2):
            for n in l_base:
                if min_l*0.75 <= multiplicator * n <= max_l * 1.2:
                    l.append(multiplicator * n)
            multiplicator *= 10
    return l

