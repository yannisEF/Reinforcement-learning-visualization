clear
echo Training the model...
echo
python trainModel.py --env Pendulum-v1 --max_learn 10000 --save_freq 500 --save_path Models --name_prefix example_pendulum
clear
echo Preparing the input policies...
echo
python preparePolicies.py --inputFolder Models --inputNames "example_pendulum_4000_steps; example_pendulum_4500_steps; example_pendulum_5500_steps; example_pendulum_6000_steps" --outputName "example_around_5000"
clear
echo Computing the Vignette...
echo
python Vignette.py --env Pendulum-v1 --inputFolder Models --inputName example_pendulum_5000_steps --eval_maxiter 5 --nb_lines 10 --policiesPath ComparePolicies/example_around_5000.xz --outputName example_pendulum_5000_steps
clear
echo Loading the computed Vignette...
echo
python savedVignette.py --directory SavedVignette --filename example_pendulum_5000_steps
clear
