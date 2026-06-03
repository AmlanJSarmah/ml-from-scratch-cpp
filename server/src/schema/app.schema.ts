import { z } from 'zod';

export const benchMarkQuery = z.object({
  model: z.enum([
    'linear_regression',
    'logistic_regression',
    'naive_bayes',
    'k_means',
  ]),
  dataset: z.enum([
    'iris',
    'california_housing',
    'breast_cancer',
    'titanic_survived',
  ]),
});
