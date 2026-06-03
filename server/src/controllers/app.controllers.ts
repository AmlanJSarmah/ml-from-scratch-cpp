import { Request, Response, NextFunction } from 'express';
import { benchMarkQuery } from '../schema/app.schema.js';

export const getBenchmarks = (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  const benchmarkParameters = {
    model: req.query.model,
    dataset: req.query.dataset,
  };
  try {
    const queries = benchMarkQuery.parse(benchmarkParameters);
    console.log(queries);
    res.status(200).send({ message: 'Success!' });
  } catch (err) {
    next(err);
  }
};
