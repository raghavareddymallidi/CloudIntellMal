import fs from 'fs';
import path from 'path';

export default function handler(req, res) {
    // Set CORS headers
    res.setHeader('Access-Control-Allow-Credentials', 'true');
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS');
    res.setHeader(
        'Access-Control-Allow-Headers',
        'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version'
    );

    if (req.method === 'OPTIONS') {
        res.status(200).end();
        return;
    }

    const { type } = req.query;
    const fileMapping = {
        performance: 'tcn_evaluation_results.json',
        comparison: 'model_comparison.json',
        noise: 'noise_results.json',
        protect90: 'protect90_analysis.json',
        ieee13: 'ieee13_analysis.json',
        history: 'training_history.json',
        cross_system: 'cross_system_results.json'
    };

    const fileName = fileMapping[type];
    if (!fileName) {
        return res.status(404).json({ detail: `Result endpoint '${type}' not found.` });
    }

    try {
        const filePath = path.join(process.cwd(), 'results', fileName);
        if (!fs.existsSync(filePath)) {
            return res.status(404).json({ detail: `Result file '${fileName}' not found.` });
        }
        const data = fs.readFileSync(filePath, 'utf-8');
        res.setHeader('Content-Type', 'application/json');
        return res.status(200).send(data);
    } catch (error) {
        console.error(`Error reading results/${fileName}:`, error);
        return res.status(500).json({ detail: 'Failed to read evaluation results.' });
    }
}

