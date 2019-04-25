rm eddieMDPs/*
rm eddieStartMDPs/*
rm eddieConfigs/*

cp mdps/medium-factory.json eddieMDPs/medium-factory.json
cp startMDPs/factory-minimal.json eddieStartMDPs/factory-minimal.json

cp configs/default.json eddieConfigs/default.json
cp configs/lowTolerance.json eddieConfigs/lowTolerance.json
cp configs/highTolerance.json eddieConfigs/highTolerance.json
cp configs/nonConservative.json eddieConfigs/nonConservative.json
