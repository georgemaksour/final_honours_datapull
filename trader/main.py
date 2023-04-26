from optimiser import *

if __name__ == '__main__':
    if DEEP_REINFORCEMENT_LEARNING:

        # Test for hyper-parameters
        optimise_agent_wrapper()

        # This is for looking at the reliability of results
        # record_over_time()

        # This is for looking at the average and standard deviation of profit metrics
        # multiple_runs_wrapper()

        # For single run to get convergence plots
        # single_run_for_diagnostics()

        # For single run to get final results
        # single_model()

    if MACHINE_LEARNING:
        machine_learning_simulation(NEURAL_NETWORK)


