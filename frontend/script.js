class App {
  constructor() {
    const serverHost = window.location.hostname;
    this.ws = new WebSocket(`ws://${serverHost}:8765`);
    this.canvas = document.getElementById("sim-canvas");
    this.ctx = this.canvas.getContext("2d");

    this.size = 8;
    this.cellSize = 60;
    this.state = null;

    this.setupWebsocket();
    this.drawEmptyBoard();
  }

  setupWebsocket() {
    this.ws.onopen = () => this.ws.send(JSON.stringify({ client: "frontend" }));

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "update") {
        this.state = data;

        document.getElementById("p1-status").innerText = data.p1_connected
          ? "Player 1 (Black)"
          : "Player 1 (Disconnected)";
        document.getElementById("p2-status").innerText = data.p2_connected
          ? "Player 2 (White)"
          : "Player 2 (Disconnected)";

        document.getElementById("p1-discs").innerText = data.disc_counts[1];
        document.getElementById("p2-discs").innerText = data.disc_counts[2];
        document.getElementById("p1-score").innerText = data.match_scores[1];
        document.getElementById("p2-score").innerText = data.match_scores[2];

        const turnText = document.getElementById("turn-indicator");
        if (data.p1_connected && data.p2_connected) {
          turnText.innerText = `Current Turn: Player ${data.current_turn}`;
          turnText.style.color =
            data.current_turn === 1 ? "#2E3440" : "#ECEFF4";
        } else {
          turnText.innerText = "Waiting for agents...";
          turnText.style.color = "#D8DEE9";
        }

        this.draw();
      }
    };
  }

  drawEmptyBoard() {
    this.ctx.fillStyle = "#A3BE8C"; // Nord Green felt
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.strokeStyle = "#434C5E"; // Dark grid lines
    this.ctx.lineWidth = 2;

    for (let i = 0; i <= this.size; i++) {
      this.ctx.beginPath();
      this.ctx.moveTo(i * this.cellSize, 0);
      this.ctx.lineTo(i * this.cellSize, this.canvas.height);
      this.ctx.stroke();
      this.ctx.beginPath();
      this.ctx.moveTo(0, i * this.cellSize);
      this.ctx.lineTo(this.canvas.width, i * this.cellSize);
      this.ctx.stroke();
    }
  }

  draw() {
    if (!this.state || !this.state.board) return;
    this.drawEmptyBoard();

    const board = this.state.board;
    const validActions = this.state.valid_actions || [];

    for (let y = 0; y < this.size; y++) {
      for (let x = 0; x < this.size; x++) {
        const cx = x * this.cellSize + this.cellSize / 2;
        const cy = y * this.cellSize + this.cellSize / 2;

        // Draw Discs
        if (board[y][x] !== 0) {
          this.ctx.beginPath();
          this.ctx.arc(cx, cy, this.cellSize / 2.5, 0, Math.PI * 2);
          this.ctx.fillStyle = board[y][x] === 1 ? "#2E3440" : "#ECEFF4"; // Black or White
          this.ctx.fill();

          // 3D Rim effect
          this.ctx.lineWidth = 3;
          this.ctx.strokeStyle = board[y][x] === 1 ? "#1b1e25" : "#D8DEE9";
          this.ctx.stroke();
        }

        // Draw Valid Move Indicators (small dots)
        const isValid = validActions.some(
          (action) => action[0] === x && action[1] === y,
        );
        if (isValid) {
          this.ctx.beginPath();
          this.ctx.arc(cx, cy, 6, 0, Math.PI * 2);
          this.ctx.fillStyle =
            this.state.current_turn === 1
              ? "rgba(46,52,64,0.3)"
              : "rgba(236,239,244,0.5)";
          this.ctx.fill();
        }
      }
    }
  }
}
const app = new App();
