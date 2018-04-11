from docplex.cp.model import *
from docplex.cp.expression import _FLOATING_POINT_PRECISION
import time

import pandas as pd
import numpy as np


schedUnitPerDurationUnit = 1  # DurationUnit is minutes
duration_units_per_day = 1440.0

# Define global constants for date to integer conversions
horizon_start_date = pd.to_datetime('Wed Apr 11 00:00:00 UTC 2018')
horizon_end_date = pd.to_datetime('Sat Apr 08 00:00:00 UTC 2028')
nanosecs_per_sec = 1000.0 * 1000 * 1000
secs_per_day = 3600.0 * 24

# Convert type to 'int64'
def helper_int64_convert(arg):
    if pd.__version__ < '0.20.0':
        return arg.astype('int64', raise_on_error=False)
    else:
        return arg.astype('int64', errors='ignore')

# Parse and convert an integer Series to a date Series
# Integer value represents the number of schedule units (time granularity for engine) since horizon start
def helper_convert_int_series_to_date(sched_int_series):
    return pd.to_datetime(sched_int_series * secs_per_day / duration_units_per_day / schedUnitPerDurationUnit * nanosecs_per_sec + horizon_start_date.value, errors='coerce')

# Return index values of a multi-index from index name
def helper_get_level_values(df, column_name):
    return df.index.get_level_values(df.index.names.index(column_name))

# Convert a duration Series to a Series representing the number of scheduling units
def helper_convert_duration_series_to_scheduling_unit(duration_series, nb_input_data_units_per_day):
    return helper_int64_convert(duration_series * duration_units_per_day * schedUnitPerDurationUnit / nb_input_data_units_per_day)

# Label constraint
expr_counter = 1
def helper_add_labeled_cpo_constraint(mdl, expr, label, context=None, columns=None):
    global expr_counter
    if isinstance(expr, bool):
        pass  # Adding a trivial constraint: if infeasible, docplex will raise an exception it is added to the model
    else:
        expr.name = '_L_EXPR_' + str(expr_counter)
        expr_counter += 1
        if columns:
            ctxt = ", ".join(str(getattr(context, col)) for col in columns)
        else:
            if context:
                ctxt = context.Index if isinstance(context.Index, str) is not None else ", ".join(context.Index)
            else:
                ctxt = None
        expr_to_info[expr.name] = (label, ctxt)
    mdl.add(expr)



# Data model definition for each table
# Data collection: list_of_Operation ['duration', 'id', 'technician', 'type']
# Data collection: list_of_Technician ['Id']
# Data collection: list_of_Type ['id']

# Create a pandas Dataframe for each data table
list_of_Operation = inputs['operation']
list_of_Operation = list_of_Operation[['duration', 'id', 'technician', 'type']].copy()
list_of_Operation.rename(columns={'duration': 'duration', 'id': 'id', 'technician': 'technician', 'type': 'type'}, inplace=True)
list_of_Technician = inputs['technician']
list_of_Technician = list_of_Technician[['Id']].copy()
list_of_Technician.rename(columns={'Id': 'Id'}, inplace=True)
# --- Handling table for implicit concept
list_of_Type = pd.DataFrame(inputs['operation']['type'].unique(), columns=['id']).dropna()

# Convert all input durations to internal time unit
list_of_Operation['duration'] = helper_convert_duration_series_to_scheduling_unit(list_of_Operation.duration, 1440.0)

# Set index when a primary key is defined
list_of_Operation.set_index('id', inplace=True)
list_of_Operation.sort_index(inplace=True)
list_of_Operation.index.name = 'id_of_Operation'
list_of_Technician.set_index('Id', inplace=True)
list_of_Technician.sort_index(inplace=True)
list_of_Technician.index.name = 'id_of_Technician'
list_of_Type.set_index('id', inplace=True)
list_of_Type.sort_index(inplace=True)
list_of_Type.index.name = 'id_of_Type'

# Create data frame as cartesian product of: Operation x Technician
list_of_SchedulingAssignment = pd.DataFrame(index=pd.MultiIndex.from_product((list_of_Operation.index, list_of_Technician.index), names=['id_of_Operation', 'id_of_Technician']))


def build_model():
    mdl = CpoModel()

    # Definition of model variables
    list_of_SchedulingAssignment['interval'] = interval_var_list(len(list_of_SchedulingAssignment), end=(INTERVAL_MIN, INTERVAL_MAX / 4), optional=True)
    list_of_SchedulingAssignment['schedulingAssignmentVar'] = list_of_SchedulingAssignment.interval.apply(mdl.presence_of)
    list_of_Operation['interval'] = interval_var_list(len(list_of_Operation), end=(INTERVAL_MIN, INTERVAL_MAX / 4), optional=True)
    list_of_Operation['taskStartVar'] = list_of_Operation.interval.apply(mdl.start_of)
    list_of_Operation['taskEndVar'] = list_of_Operation.interval.apply(mdl.end_of)
    list_of_Operation['taskDurationVar'] = list_of_Operation.interval.apply(mdl.size_of)
    list_of_SchedulingAssignment['taskAssignmentDurationVar'] = list_of_SchedulingAssignment.interval.apply(mdl.size_of)
    list_of_Operation['taskAbsenceVar'] = 1 - list_of_Operation.interval.apply(mdl.presence_of)
    list_of_Operation['taskPresenceVar'] = list_of_Operation.interval.apply(mdl.presence_of)


    # Definition of model
    # Objective cMinimizeMakespan-
    # Combine weighted criteria: 
    # 	cMinimizeMakespan cMinimizeMakespan{
    # 	cScaledGoal.scaleFactorExpr = 1,
    # 	cSingleCriterionGoal.goalFilter = null,
    # 	cMinimizeMakespan.taskEnd = cTaskEnd[operation],
    # 	cSingleCriterionGoal.numericExpr = max of count( cTaskEnd[operation]) over cTaskEnd[operation],
    # 	cMinimizeMakespan.task = operation} with weight 5.0
    agg_Operation_taskEndVar_SG1 = mdl.max(list_of_Operation.taskEndVar)
    
    kpi_1 = integer_var(name='kpi_1')
    mdl.add(kpi_1 >= 1.0 * (agg_Operation_taskEndVar_SG1 / schedUnitPerDurationUnit) / 1 - 1 + _FLOATING_POINT_PRECISION)
    mdl.add(kpi_1 <= 1.0 * (agg_Operation_taskEndVar_SG1 / schedUnitPerDurationUnit) / 1)
    mdl.add_kpi(kpi_1, name='time to complete all operations')
    
    mdl.add(minimize( 0
        # Sub Goal cMinimizeMakespan_cMinimizeGoal
        # Minimize time to complete all operations
        + 1.0 * (agg_Operation_taskEndVar_SG1 / schedUnitPerDurationUnit) / 1
    ))
    
    # [ST_1] Constraint : cLimitNumberOfResourcesAssignedToEachActivitySched_cIterativeRelationalConstraint
    # The number of technician assignments for each operation is equal to 1
    # Label: CT_1_The_number_of_technician_assignments_for_each_operation_is_equal_to_1
    join_Operation_SchedulingAssignment = list_of_Operation.join(list_of_SchedulingAssignment, rsuffix='_right', how='inner')
    groupbyLevels = [join_Operation_SchedulingAssignment.index.names.index(name) for name in list_of_Operation.index.names]
    groupby_Operation_SchedulingAssignment = join_Operation_SchedulingAssignment.schedulingAssignmentVar.groupby(level=groupbyLevels).sum().to_frame()
    for row in groupby_Operation_SchedulingAssignment.itertuples(index=True):
        helper_add_labeled_cpo_constraint(mdl, row.schedulingAssignmentVar == 1, 'The number of technician assignments for each operation is equal to 1', row)
    
    # [ST_2] Constraint : cSetFixedDurationSpezProp_cIterativeRelationalConstraint
    # The schedule must respect the duration specified for each operation
    # Label: CT_2_The_schedule_must_respect_the_duration_specified_for_each_operation
    for row in list_of_Operation[list_of_Operation.duration.notnull()].itertuples(index=True):
        helper_add_labeled_cpo_constraint(mdl, size_of(row.interval, int(row.duration)) == int(row.duration), 'The schedule must respect the duration specified for each operation', row)
    
    # [ST_3] Constraint : cForceTaskPresence_cIterativeRelationalConstraint
    # All operations are present
    # Label: CT_3_All_operations_are_present
    for row in list_of_Operation.itertuples(index=True):
        helper_add_labeled_cpo_constraint(mdl, row.taskAbsenceVar != 1, 'All operations are present', row)
    
    # [ST_4] Constraint : cTaskIsFirst_cTaskIsFirst
    # Each assigned operation where type is start must be performed first by assigned technicians
    # Label: CT_4_Each_assigned_operation_where_type_is_start_must_be_performed_first_by_assigned_technicians
    groupbyLevels = [list_of_SchedulingAssignment.index.names.index(name) for name in list_of_Technician.index.names]
    groupby_SchedulingAssignment = list_of_SchedulingAssignment.interval.groupby(level=groupbyLevels).apply(list).to_frame(name='interval')
    list_of_Technician['sequence_var'] = groupby_SchedulingAssignment.apply(lambda row: sequence_var(row.interval), axis=1)
    filtered_Operation = list_of_Operation[list_of_Operation.type == 'start'].copy()
    join_SchedulingAssignment_Operation = list_of_SchedulingAssignment.join(filtered_Operation.type, how='inner')
    join_SchedulingAssignment_Operation_Technician = join_SchedulingAssignment_Operation.join(list_of_Technician.sequence_var, how='inner')
    for row in join_SchedulingAssignment_Operation_Technician.itertuples(index=True):
        helper_add_labeled_cpo_constraint(mdl, first(row.sequence_var, row.interval), 'Each assigned operation where type is start must be performed first by assigned technicians', row)
    
    # [ST_5] Constraint : cTaskIsLast_cTaskIsLast
    # Each assigned operation where type is end must be performed last by assigned technicians
    # Label: CT_5_Each_assigned_operation_where_type_is_end_must_be_performed_last_by_assigned_technicians
    filtered_Operation = list_of_Operation[list_of_Operation.type == 'end'].copy()
    join_SchedulingAssignment_Operation = list_of_SchedulingAssignment.join(filtered_Operation.type, how='inner')
    join_SchedulingAssignment_Operation_Technician = join_SchedulingAssignment_Operation.join(list_of_Technician.sequence_var, how='inner')
    for row in join_SchedulingAssignment_Operation_Technician.itertuples(index=True):
        helper_add_labeled_cpo_constraint(mdl, last(row.sequence_var, row.interval), 'Each assigned operation where type is end must be performed last by assigned technicians', row)
    
    # [ST_6] Constraint : cSchedulingAssignmentCompatibilityConstraintOnPair_cCategoryCompatibilityConstraintOnPair
    # For each technician to operation assignment, assigned technician includes technician of assigned operation
    # Label: CT_6_For_each_technician_to_operation_assignment__assigned_technician_includes_technician_of_assigned_operation
    join_SchedulingAssignment_Operation = list_of_SchedulingAssignment.join(list_of_Operation.technician, how='inner')
    filtered_SchedulingAssignment_Operation = join_SchedulingAssignment_Operation.loc[helper_get_level_values(join_SchedulingAssignment_Operation, 'id_of_Technician') == join_SchedulingAssignment_Operation.technician].copy()
    helper_add_labeled_cpo_constraint(mdl, mdl.sum(join_SchedulingAssignment_Operation.schedulingAssignmentVar[(join_SchedulingAssignment_Operation.technician.notnull()) & (~join_SchedulingAssignment_Operation.index.isin(filtered_SchedulingAssignment_Operation.index.values))]) == 0, 'For each technician to operation assignment, assigned technician includes technician of assigned operation')
    
    # Scheduling internal structure
    groupbyLevels = [list_of_SchedulingAssignment.index.names.index(name) for name in list_of_Operation.index.names]
    groupby_SchedulingAssignment = list_of_SchedulingAssignment.interval.groupby(level=groupbyLevels).apply(list).to_frame(name='interval')
    join_SchedulingAssignment_Operation = groupby_SchedulingAssignment.join(list_of_Operation.interval, rsuffix='_right', how='inner')
    for row in join_SchedulingAssignment_Operation.itertuples(index=False):
        mdl.add(synchronize(row.interval_right, row.interval))
    
    # link presence if not alternative
    groupbyLevels = [list_of_SchedulingAssignment.index.names.index(name) for name in list_of_Operation.index.names]
    groupby_SchedulingAssignment = list_of_SchedulingAssignment.schedulingAssignmentVar.groupby(level=groupbyLevels).agg(lambda l: mdl.max(l.tolist())).to_frame()
    join_SchedulingAssignment_Operation = groupby_SchedulingAssignment.join(list_of_Operation.taskPresenceVar, how='inner')
    for row in join_SchedulingAssignment_Operation.itertuples(index=False):
        mdl.add(row.schedulingAssignmentVar <= row.taskPresenceVar)
    
    # no overlap
    for row in list_of_Technician.reset_index().itertuples(index=False):
        mdl.add(no_overlap(row.sequence_var))


    return mdl


def solve_model(mdl):
    params = CpoParameters()
    params.TimeLimit = 20
    solver = CpoSolver(mdl, params=params, trace_log=True)
    try:
        for i, msol in enumerate(solver):
            ovals = msol.get_objective_values()
            print("Objective values: {}".format(ovals))
            for k, v in msol.get_kpis().iteritems():
                print k, '-->', v
            export_solution(msol)
            if ovals is None:
                break  # No objective: stop after first solution
        # If model is infeasible, invoke conflict refiner to return
        if solver.get_last_solution().get_solve_status() == SOLVE_STATUS_INFEASIBLE:
            conflicts = solver.refine_conflict()
            export_conflicts(conflicts)
    except CpoException as e:
        # Solve has been aborted from an external action
        print('An exception has been raised: %s' % str(e))
        raise e


expr_to_info = {}


def export_conflicts(conflicts):
    # Display conflicts in console
    print conflicts
    list_of_conflicts = pd.DataFrame(columns=['constraint', 'context', 'detail'])
    for item, index in zip(conflicts.member_constraints, range(len(conflicts.member_constraints))):
        label, context = expr_to_info.get(item.name, ('N/A', item.name))
        constraint_detail = expression._to_string(item)
        # Print conflict information in console
        print("Conflict involving constraint: %s, \tfor: %s -> %s" % (label, context, constraint_detail))
        list_of_conflicts = list_of_conflicts.append(pd.DataFrame({'constraint': label, 'context': str(context), 'detail': constraint_detail},
                                                                  index=[index], columns=['constraint', 'context', 'detail']))

    # Update of the ``outputs`` dict must take the 'Lock' to make this action atomic,
    # in case the job is aborted
    global output_lock
    with output_lock:
        outputs['list_of_conflicts'] = list_of_conflicts


def export_solution(msol):
    start_time = time.time()
    list_of_SchedulingAssignment_solution = pd.DataFrame(index=list_of_SchedulingAssignment.index)
    list_of_SchedulingAssignment_solution['schedulingAssignmentVar'] = list_of_SchedulingAssignment.interval.apply(lambda r: (1 if msol.solution.get_var_solution(r).is_present() else 0) if msol.solution.get_var_solution(r) else np.NaN)
    list_of_Operation_solution = pd.DataFrame(index=list_of_Operation.index)
    list_of_Operation_solution = list_of_Operation_solution.join(pd.DataFrame([msol.solution[interval] if msol.solution[interval] else (None, None, None) for interval in list_of_Operation.interval], index=list_of_Operation.index, columns=['taskStartVar', 'taskEndVar', 'taskDurationVar']))
    list_of_Operation_solution['taskStartVarDate'] = helper_convert_int_series_to_date(list_of_Operation_solution.taskStartVar)
    list_of_Operation_solution['taskEndVarDate'] = helper_convert_int_series_to_date(list_of_Operation_solution.taskEndVar)
    list_of_Operation_solution.taskStartVar /= schedUnitPerDurationUnit
    list_of_Operation_solution.taskEndVar /= schedUnitPerDurationUnit
    list_of_Operation_solution.taskDurationVar /= schedUnitPerDurationUnit
    list_of_SchedulingAssignment_solution['taskAssignmentDurationVar'] = list_of_SchedulingAssignment.interval.apply(lambda r: msol.solution.get_var_solution(r).get_size() if msol.solution.get_var_solution(r) else np.NaN)
    list_of_SchedulingAssignment_solution.taskAssignmentDurationVar /= schedUnitPerDurationUnit
    list_of_Operation_solution['taskAbsenceVar'] = list_of_Operation.interval.apply(lambda r: (1 if msol.solution.get_var_solution(r).is_absent() else 0) if msol.solution.get_var_solution(r) else np.NaN)
    list_of_Operation_solution['taskPresenceVar'] = list_of_Operation.interval.apply(lambda r: (1 if msol.solution.get_var_solution(r).is_present() else 0) if msol.solution.get_var_solution(r) else np.NaN)

    # Filter rows for non-selected assignments
    list_of_SchedulingAssignment_solution = list_of_SchedulingAssignment_solution[list_of_SchedulingAssignment_solution.schedulingAssignmentVar > 0.5]

    # Update of the ``outputs`` dict must take the 'Lock' to make this action atomic,
    # in case the job is aborted
    global output_lock
    with output_lock:
        outputs['list_of_Operation_solution'] = list_of_Operation_solution.reset_index()
        outputs['list_of_SchedulingAssignment_solution'] = list_of_SchedulingAssignment_solution.reset_index()

    elapsed_time = time.time() - start_time
    print('solution export done in ' + str(elapsed_time) + ' secs')
    return


print('* building wado model')
start_time = time.time()
model = build_model()
elapsed_time = time.time() - start_time
print('model building done in ' + str(elapsed_time) + ' secs')

print('* running wado model')
start_time = time.time()
solve_model(model)
elapsed_time = time.time() - start_time
print('solve + export of all intermediate solutions done in ' + str(elapsed_time) + ' secs')
