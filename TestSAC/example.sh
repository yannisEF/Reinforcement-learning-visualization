clear
echo Training the model...
echo
python3 trainModel.py --env Pendulum-v0 --max_learn 10000 --save_freq 500 --save_path Models --name_prefix example_pendulum
clear
echo Preparing the input policies...
echo
python3 preparePolicies.py --inputFolder Models --inputNames "example_pendulum_7000_steps; example_pendulum_7500_steps; example_pendulum_8500_steps; example_pendulum_9000_steps" --outputName "example_around_8000"
clear
echo Computing the Vignette...
echo
python3 Vignette.py --env Pendulum-v0 --inputDir Models --basename example_pendulum_ --min_iter 8000 --max_iter 8001 --step_iter 500 --eval_maxiter 1 --nb_lines 10 --policiesPath ComparePolicies/example_around_8000.xz
clear
echo Loading the computed Vignette...
echo
python3 savedVignette.py --directory SavedVignette --filename example_pendulum_8000_steps
clear
