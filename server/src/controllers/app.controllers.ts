import { Request, Response } from 'express';

export const getBenchmarks = (req: Request, res: Response) => {
  res.status(200).send({ message: 'Success!' });
};
