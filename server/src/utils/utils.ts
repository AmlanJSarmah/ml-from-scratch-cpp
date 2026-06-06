type Models =
  | 'linear_regression'
  | 'logistic_regression'
  | 'k_means'
  | 'naive_bayes';

type Datasets =
  | 'iris'
  | 'california_housing'
  | 'breast_cancer'
  | 'titanic_survived';

type ClusteringReport = {
  accuracy: number;
};

type RegressionReport = {
  accuracy: number;
  r2: number;
  rmse: number;
  mae: number;
  mse: number;
};

type ClassificationReport = {
  accuracy: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  precision: number;
  recall: number;
  f1: number;
};

// TODO: Avoid classification and clustering algorithms to accept continuous datasets like california_housing
export function parseCommand(model: Models, dataset: Datasets) {
  switch (dataset) {
    case 'iris':
      if (model === 'k_means')
        return `../build/${model} ../data/iris.csv 5 1 3 1`;
      return `../build/${model} ../data/iris.csv 5 1 1`;

    case 'california_housing':
      if (model === 'k_means')
        return `../build/k_means ../data/california_housing.csv 9 0 2 1`;
      return `../build/${model} ../data/california_housing.csv 9 0 1`;

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

export function parseCommandBenchmark(model: Models, dataset: Datasets) {
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

export function generateReport(model: Models, result: string) {
  if (model === 'k_means') {
    const resultSplit = result.split('\n');
    const accuracy = parseFloat(resultSplit[0]);
    return { accuracy: accuracy } as ClusteringReport;
  } else if (model === 'linear_regression') {
    const resultSplit = result.split('\n');
    const parsedResult: RegressionReport = {
      accuracy: parseFloat(resultSplit[0]),
      r2: parseFloat(resultSplit[1]),
      rmse: parseFloat(resultSplit[2]),
      mae: parseFloat(resultSplit[3]),
      mse: parseFloat(resultSplit[4]),
    };
    return parsedResult;
  } else {
    const resultSplit = result.split('\n');
    const parsedResult: ClassificationReport = {
      accuracy: parseFloat(resultSplit[0]),
      tp: parseFloat(resultSplit[1]),
      fp: parseFloat(resultSplit[2]),
      fn: parseFloat(resultSplit[3]),
      tn: parseFloat(resultSplit[4]),
      precision: parseFloat(resultSplit[5]),
      recall: parseFloat(resultSplit[6]),
      f1: parseFloat(resultSplit[7]),
    };
    return parsedResult;
  }
}
