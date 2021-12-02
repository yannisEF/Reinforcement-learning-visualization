clear
echo Training the model...
echo
python trainModel.py --env Pendulum-v1 --max_learn 10000 --save_freq 500 --save_path Models --name_prefix example_pendulum
clear
echo Preparing the input policies...
echo
python preparePolicies.py --inputFolder Models --inputNames "example_pendulum_500_steps; example_pendulum_1000_steps; example_pendulum_1500_steps; example_pendulum_2000_steps; example_pendulum_2500_steps; example_pendulum_3000_steps; example_pendulum_3500_steps; example_pendulum_4000_steps; example_pendulum_4500_steps; example_pendulum_5500_steps; example_pendulum_6000_steps; example_pendulum_6500_steps; example_pendulum_7000_steps; example_pendulum_7500_steps; example_pendulum_8000_steps; example_pendulum_8500_steps; example_pendulum_9000_steps; example_pendulum_9500_steps; example_pendulum_10000_steps" --outputName "example_around_5000"
clear
echo Computing the Vignette...
echo
python Vignette.py --env Pendulum-v1 --inputFolder Models --inputName example_pendulum_5000_steps --eval_maxiter 5 --nb_lines 50 --policiesPath ComparePolicies/example_around_5000.xz --maxalpha 100 --stepalpha 0.5 --outputName example_pendulum_5000_steps_large
clear
echo Loading the computed Vignette...
echo
python savedVignette.py --directory SavedVignette --filename example_pendulum_5000_steps_large --darkBg True --rotate False
clear
