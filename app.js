// app.js
class QuickDrawApp {
    constructor() {
        this.canvas = document.getElementById('drawCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.isDrawing = false;
        this.model = null;
        this.classes = [];
        this.currentPrompt = '';
        this.timeLeft = 30; 
        this.timerInterval = null;
        
        this.setupCanvas();
        this.setupEventListeners();
        this.initModel();
    }

    setupCanvas() {
        this.ctx.lineWidth = 24;
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
        this.ctx.strokeStyle = '#2D3142';
        this.clearCanvas();
    }

    setupEventListeners() {
        this.canvas.addEventListener('mousedown', (e) => this.startDrawing(e));
        this.canvas.addEventListener('mousemove', (e) => this.draw(e));
        this.canvas.addEventListener('mouseup', () => this.stopDrawing());
        this.canvas.addEventListener('mouseout', () => this.stopDrawing());
        
        this.canvas.addEventListener('touchstart', (e) => { e.preventDefault(); this.startDrawing(e.touches[0]); });
        this.canvas.addEventListener('touchmove', (e) => { e.preventDefault(); this.draw(e.touches[0]); });
        this.canvas.addEventListener('touchend', () => this.stopDrawing());
        
        document.getElementById('clearBtn').addEventListener('click', () => {
            this.clearCanvas();
            this.displayPredictions([{ name: 'Start drawing...', confidence: 0 }]);
        });
        document.getElementById('newPromptBtn').addEventListener('click', () => this.newPrompt());
    }

    getCoordinates(e) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        };
    }

    startDrawing(e) {
        this.isDrawing = true;
        const coords = this.getCoordinates(e);
        this.ctx.beginPath();
        this.ctx.moveTo(coords.x, coords.y);
    }

    draw(e) {
        if (!this.isDrawing) return;
        const coords = this.getCoordinates(e);
        this.ctx.lineTo(coords.x, coords.y);
        this.ctx.stroke();
    }

    stopDrawing() { 
        if(this.isDrawing) {
            this.isDrawing = false; 
            this.predict();
        }
    }

    clearCanvas() {
        this.ctx.fillStyle = '#FFFFFF';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }

    newPrompt() {
        if (this.classes.length === 0) return;
        this.currentPrompt = this.classes[Math.floor(Math.random() * this.classes.length)];
        document.getElementById('currentPrompt').textContent = this.currentPrompt;
        this.clearCanvas();
        this.displayPredictions([{ name: 'Start drawing...', confidence: 0 }]);
        this.resetTimer();
    }

    resetTimer() {
        if (this.timerInterval) clearInterval(this.timerInterval);
        this.timeLeft = 30; 
        document.getElementById('timer').textContent = `${this.timeLeft}s`;
        
        this.timerInterval = setInterval(() => {
            this.timeLeft--;
            document.getElementById('timer').textContent = `${this.timeLeft}s`;
            if (this.timeLeft <= 0) {
                clearInterval(this.timerInterval);
                this.predict();
            }
        }, 1000);
    }

    async initModel() {
        try {
            this.model = await tf.loadLayersModel('./web_model/model.json');
            const response = await fetch('./web_model/classes.json');
            this.classes = await response.json();
            this.newPrompt();
        } catch (error) {
            console.error('Failed to load model:', error);
        }
    }

    // preprocessCanvas() {
    //     const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
    //     const data = imageData.data;
    //     const size = 28;
    //     const processed = new Array(size * size).fill(0);
    //     const scaleX = this.canvas.width / size;
    //     const scaleY = this.canvas.height / size;
        
    //     for (let y = 0; y < size; y++) {
    //         for (let x = 0; x < size; x++) {
    //             let sum = 0;
    //             let count = 0;
    //             for (let dy = 0; dy < scaleY; dy++) {
    //                 for (let dx = 0; dx < scaleX; dx++) {
    //                     const sx = Math.floor(x * scaleX + dx);
    //                     const sy = Math.floor(y * scaleY + dy);
    //                     const idx = (sy * this.canvas.width + sx) * 4;
    //                     const gray = 255 - (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
    //                     sum += gray;
    //                     count++;
    //                 }
    //             }
    //             processed[y * size + x] = sum / count;
    //         }
    //     }
    //     return processed;
    // }
    preprocessCanvas() {
        const size = 28;
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = size;
        tempCanvas.height = size;

        // 1. Find the Bounding Box (where the ink actually is)
        const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        const data = imageData.data;
        let minX = this.canvas.width, minY = this.canvas.height, maxX = 0, maxY = 0;
        let found = false;

        for (let y = 0; y < this.canvas.height; y++) {
            for (let x = 0; x < this.canvas.width; x++) {
                const i = (y * this.canvas.width + x) * 4;
                if (data[i] < 200) { // If pixel is not white (ink found)
                    if (x < minX) minX = x;
                    if (y < minY) minY = y;
                    if (x > maxX) maxX = x;
                    if (y > maxY) maxY = y;
                    found = true;
                }
            }
        }

        // 2. If canvas is empty, return zeros
        if (!found) return new Array(size * size).fill(0);

        // 3. Crop and Center the drawing into the 28x28 box
        const width = maxX - minX;
        const height = maxY - minY;
        const maxDim = Math.max(width, height);
        
        // Add a small padding and draw it centered onto the 28x28 temp canvas
        tempCtx.fillStyle = "white";
        tempCtx.fillRect(0, 0, size, size);
        tempCtx.drawImage(this.canvas, minX, minY, width, height, 2, 2, size - 4, size - 4);

        // 4. Convert to 0-255 grayscale
        const finalData = tempCtx.getImageData(0, 0, size, size).data;
        const processed = new Array(size * size);
        for (let i = 0; i < size * size; i++) {
            processed[i] = 255 - (finalData[i * 4] + finalData[i * 4 + 1] + finalData[i * 4 + 2]) / 3;
        }
        return processed;
    }

    async predict() {
        if (!this.model) return;
        
        const imageData = this.preprocessCanvas();
        // --- ADD THIS LOG ---
        // This will tell us if the AI sees anything at all.
        // If it prints '0', the AI is seeing a blank screen!
        const maxVal = Math.max(...imageData);
        console.log("AI sees a max pixel value of:", maxVal);
        try {
            const tensor = tf.tensor(imageData, [1, 28, 28, 1], 'float32');
            
            let output = this.model.predict(tensor);
            if (Array.isArray(output)) output = output[0];
            if (output.constructor.name === 'Object') output = Object.values(output)[0];

            const predictions = await output.data();
            tensor.dispose();
            
            const results = Array.from(predictions)
                .map((confidence, i) => ({ name: this.classes[i], confidence }))
                .sort((a, b) => b.confidence - a.confidence)
                .slice(0, 5);
            
            this.displayPredictions(results);

        } catch (err) {
            console.error("Prediction error:", err);
            const list = document.getElementById('predictions');
            list.innerHTML = `
                <li class="prediction-item" style="background: #ffebee; border-color: #c62828;">
                    <span class="prediction-name" style="color: #c62828; font-size: 0.85rem;">
                        Bug caught: ${err.message}
                    </span>
                </li>`;
        }
    }

    displayPredictions(predictions) {
        const list = document.getElementById('predictions');
        if (predictions[0].confidence === 0) {
            list.innerHTML = `<li class="prediction-item"><span class="prediction-name">${predictions[0].name}</span></li>`;
            return;
        }

        list.innerHTML = predictions.map(p => `
            <li class="prediction-item">
                <span class="prediction-name">${p.name}</span>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${p.confidence * 100}%"></div>
                </div>
                <span class="confidence-value">${(p.confidence * 100).toFixed(1)}%</span>
            </li>
        `).join('');
    }
}

async function init() {
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.11.0/dist/tf.min.js';
    script.onload = () => { window.app = new QuickDrawApp(); };
    document.head.appendChild(script);
}

init();