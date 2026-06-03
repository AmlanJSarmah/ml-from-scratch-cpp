import express from 'express';
import { Request, Response, NextFunction } from 'express';
import { treeifyError, ZodError } from 'zod';
import appRouter from './routes/app.routes.js';

const app = express();

app.use(appRouter);

app.listen(8080, () => {
  console.log('Listening on 8080');
});

app.use((error: unknown, req: Request, res: Response, next: NextFunction) => {
  if (error instanceof ZodError) {
    return res.status(400).json({
      message: 'Validation failed',
      errors: treeifyError(error),
    });
  }
  console.error(error);
  res.status(500).json({ message: 'Internal Server Error' });
});
