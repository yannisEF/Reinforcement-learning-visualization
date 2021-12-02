clear
echo Training the model...
echo
python trainModel.py --env Pendulum-v0 --max_learn 10000 --save_freq 500 --save_path Models --name_prefix example_pendulum
clear
echo Preparing the input policies...
echo
python preparePolicies.py --inputFolder Models --inputNames "example_pendulum_7000_steps; example_pendulum_7500_steps; example_pendulum_8500_steps; example_pendulum_9000_steps" --outputName "example_around_8000"
clear
echo Computing the Vignette...
echo
python Vignette.py --env Pendulum-v0 --inputFolder Models --inputName example_pendulum_8000_steps --eval_maxiter 5 --nb_lines 10 --policiesPath ComparePolicies/example_around_8000.xz --outputName example_pendulum_8000_steps
clear
echo Loading the computed Vignette...
echo
python savedVignette.py --directory SavedVignette --filename example_pendulum_8000_steps
clear
