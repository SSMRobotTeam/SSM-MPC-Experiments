%% Main entry for the SSM experiment
projectRoot = fileparts(mfilename('fullpath'));

addpath(projectRoot);
addpath(fullfile(projectRoot, 'casadi'));
addpath(fullfile(projectRoot, 'dynamic', 'SSM'));
addpath(fullfile(projectRoot, 'dynamic', 'SSM', 'ROM8_Expe'));
addpath(fullfile(projectRoot, 'measure'));
addpath(fullfile(projectRoot, 'PlotFig'));
addpath(fullfile(projectRoot, 'utils'));
addpath(fullfile(projectRoot, 'zmq_message'));

javaaddpath(fullfile(projectRoot, 'zmq_message', 'jeromq-0.6.0.jar'));

previousFolder = pwd;
cleanupObj = onCleanup(@() cd(previousFolder));
cd(projectRoot);

run(fullfile(projectRoot, 'Main_demo_SSM.mlx'));
