rm eddieMDPs/*
rm eddieStartMDPs/*
rm eddieConfigs/*

cp mdps/coffee.json eddieMDPs/coffee.json
cp startMDPs/coffee-start-minimal.json eddieStartMDPs/coffee-start-minimal.json

cp configs/default.json eddieConfigs/default.json
cp configs/lowTolerance.json eddieConfigs/lowTolerance.json
cp configs/highTolerance.json eddieConfigs/highTolerance.json
cp configs/nonConservative.json eddieConfigs/nonConservative.json
