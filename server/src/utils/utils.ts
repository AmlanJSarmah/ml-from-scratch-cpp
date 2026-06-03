type models =
  | 'linear_regression'
  | 'logistic_regression'
  | 'k_means'
  | 'naive_bayes';

type datasets =
  | 'iris'
  | 'california_housing'
  | 'breast_cancer'
  | 'titanic_survived';

// TODO: Avoid classification and clustering algorithms to accept continuous datasets like california_housing
export function parseCommand(model: models, dataset: datasets) {
  switch (dataset) {
    case 'iris':
      if (model === 'k_means')
        return `../build/${model} ../data/iris.csv 5 1 3 1`;
      return `../build/${model} ../data/iris.csv 5 1 1`;

    case 'california_housing':
      if (model === 'k_means')
        return `../build/k_means ../data/california_housing.csv 8 0 2 1`;
      return `../build/${model} ../data/california_housing.csv 8 0 1`;

    case 'titanic_survived':
      if (model === 'k_means')
        return `../build/k_means ../data/titanic_survived.csv 8 0 2 1`;
      return `../build/${model} ../data/titanic_survived.csv 8 0 1`;

    case 'breast_cancer':
      if (model === 'k_means')
        return `../build/k_means ../data/breast_cancer.csv 31 0 2 1`;
      return `../build/${model} ../data/breast_cancer.csv 31 0 1`;
  }
}

export function parseCommandBenchmark(model: models, dataset: datasets) {
  switch (dataset) {
    case 'iris':
      return `python3 ../benchmarks/${model}_benchmark.py ../data/iris.csv`;

    case 'california_housing':
      return `python3 ../benchmarks/${model}_benchmark.py ../data/california_housing.csv`;

    case 'titanic_survived':
      return `python3 ../benchmarks/${model}_benchmark.py ../data/titanic_survived.csv`;

    case 'breast_cancer':
      return `python3 ../benchmarks/${model}_benchmark.py ../data/breast_cancer.csv`;
  }
}
