/* docs/assets/js/particles.js
   Creates a subtle animated particle field background
*/

document.addEventListener("DOMContentLoaded", () => {
  const canvas = document.createElement("canvas");
  canvas.id = "cf-particles";
  document.body.appendChild(canvas);

  const ctx = canvas.getContext("2d");
  let width, height, particles;

  function resize() {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
    particles = Array.from({ length: 80 }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3
    }));
  }

  function draw() {
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "rgba(150, 100, 255, 0.6)";
    ctx.strokeStyle = "rgba(150, 100, 255, 0.2)";

    particles.forEach(p => {
      p.x += p.vx;
      p.y += p.vy;

      if (p.x < 0 || p.x > width) p.vx *= -1;
      if (p.y < 0 || p.y > height) p.vy *= -1;

      ctx.beginPath();
      ctx.arc(p.x, p.y, 1.5, 0, Math.PI * 2);
      ctx.fill();
    });

    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 100) {
          ctx.globalAlpha = 1 - dist / 100;
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
/* --- Particle Background Layer --- */
#cf-particles {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  z-index: 0; /* keep behind main content */
  pointer-events: none;
  background: transparent;
}

/* --- Ensure MkDocs content is above it --- */
.md-main, .md-header, .md-footer, .md-content {
  position: relative;
  z-index: 1;
}

/* Optional: improve readability over animation */
.md-content {
  background-color: rgba(255, 255, 255, 0.85); /* Light mode */
}

[data-md-color-scheme="slate"] .md-content {
  background-color: rgba(20, 20, 30, 0.75); /* Dark mode */
}
        }
      }
    }

    ctx.globalAlpha = 1;
    requestAnimationFrame(draw);
  }

  window.addEventListener("resize", resize);
  resize();
  draw();
});
