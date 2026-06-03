import express from 'express';
import { getBenchmarks } from '../controllers/app.controllers.js';

const router = express.Router();

router.get('/benchmarks', getBenchmarks);

export default router;
