import express from 'express';
import appRouter from './routes/app.routes.js';

const app = express();

app.use(appRouter);

app.listen(8080, () => {
  console.log('Listening on 8080');
});
