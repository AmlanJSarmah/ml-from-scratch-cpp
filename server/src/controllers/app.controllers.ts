import { Request, Response, NextFunction } from 'express';
import { exec } from 'child_process';
import { AppError } from '../utils/error.js';
import { benchMarkQuery } from '../schema/app.schema.js';
import {
  generateReport,
  parseCommand,
  parseCommandBenchmark,
} from '../utils/utils.js';

// TODO Iterate almost 5 times and take the average before sending benchmarks data
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
    exec(
      parseCommand(queries.model, queries.dataset),
      (error, stdout, stderr) => {
        if (error)
          return next(new AppError(500, `Command Execution ${stderr}`));
        const resultsCustom = generateReport(queries.model, stdout);
        exec(
          parseCommandBenchmark(queries.model, queries.dataset),
          (error, stdout, stderr) => {
            if (error) return next(new AppError(500, stderr));
            res.status(200).send({
              custom: resultsCustom,
              sklearn: generateReport(queries.model, stdout),
            });
          }
        );
      }
    );
    // res.status(200).send({ message: 'Success!' });
  } catch (err) {
    next(err);
  }
};
