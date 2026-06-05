import express from 'express';
import { Request, Response, NextFunction } from 'express';
import { treeifyError, ZodError } from 'zod';
import cors from 'cors';
import helmet from 'helmet';
import appRouter from './routes/app.routes.js';
import { AppError } from './utils/error.js';

const app = express();

app.use(cors());
app.use(helmet());
app.use(appRouter);

const port = Number(process.env.PORT) || 8080;

app.listen(port, () => {
  console.log(`Listening on ${port}`);
});

app.use((error: unknown, req: Request, res: Response, next: NextFunction) => {
  if (error instanceof ZodError) {
    return res.status(400).json({
      message: 'Validation failed',
      errors: treeifyError(error),
    });
  }
  if (error instanceof AppError) {
    return res.status(error.statusCode).json({ errors: error.message });
  }
  console.error(error);
  res.status(500).json({ message: 'Internal Server Error' });
});
