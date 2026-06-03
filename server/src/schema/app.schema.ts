import { z } from 'zod';

export const benchMarkQuery = z.object({
  model: z.enum([
    'linear regression',
    'logistic regression',
    'naive bayes',
    'k-means',
  ]),
  dataset: z.enum([
    'iris',
    'california housing',
    'breast cancer',
    'titanic survived',
  ]),
});
