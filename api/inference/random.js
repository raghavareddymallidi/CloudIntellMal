import fs from 'fs';
import path from 'path';

export default function handler(req, res) {
    // Set CORS headers
    res.setHeader('Access-Control-Allow-Credentials', 'true');
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,PATCH,DELETE,POST,PUT');
    res.setHeader(
        'Access-Control-Allow-Headers',
        'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version'
    );

    if (req.method === 'OPTIONS') {
        res.status(200).end();
        return;
    }

    try {
        let filePath = path.join(process.cwd(), 'api', 'sample_inferences.json');
        if (!fs.existsSync(filePath)) {
            filePath = path.join(process.cwd(), 'results', 'sample_inferences.json');
        }
        if (!fs.existsSync(filePath)) {
            return res.status(404).json({ detail: 'Sample inferences pool not found.' });
        }

        const rawData = fs.readFileSync(filePath, 'utf-8');
        const samples = JSON.parse(rawData);

        const modelType = (req.query.model_type || 'TCN').toUpperCase();
        const validModel = ['TCN', 'CNN', 'LSTM'].includes(modelType) ? modelType : 'TCN';

        // Pick a random sample from the pre-evaluated dataset pool
        const randomIndex = Math.floor(Math.random() * samples.length);
        const sample = samples[randomIndex];
        const modelResult = sample.models[validModel] || sample.models['TCN'];

        const responsePayload = {
            sample_id: sample.sample_id,
            true_detection: sample.true_detection,
            true_classification: sample.true_classification,
            predicted_detection: modelResult.predicted_detection,
            predicted_classification: modelResult.predicted_classification,
            detection_confidence: modelResult.detection_confidence,
            classification_confidence: modelResult.classification_confidence,
            needs_review: modelResult.needs_review,
            waveforms: sample.waveforms,
            time_axis: sample.time_axis
        };

        res.setHeader('Content-Type', 'application/json');
        return res.status(200).json(responsePayload);
    } catch (error) {
        console.error('Error handling /api/inference/random:', error);
        return res.status(500).json({ detail: 'Internal server error while serving inference.' });
    }
}
