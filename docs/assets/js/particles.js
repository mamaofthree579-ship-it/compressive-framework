/* ==========================================================
   Compressive Framework (CF-DPF)
   Particle Background Script — assets/js/particles.js
   ========================================================== */

/* Ensure particle canvas appears behind content */
#cf-particles {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  z-index: -1;
  pointer-events: none;
}

document.addEventListener("DOMContentLoaded", () => {
  // Create the canvas
  const canvas = document.createElement("canvas");
  canvas.id = "cf-particles";
  document.body.appendChild(canvas);

  const ctx = canvas.getContext("2d");
  let particles = [];
  const particleCount = 60;
  const linkDistance = 120;
  const maxSpeed = 0.4;

  // Adjust colors based on theme
  const getColor = () => {
    const dark = document.documentElement.getAttribute("data-md-color-scheme") === "slate";
    return dark
      ? { dot: "#BBDEFB", link: "rgba(66,165,245,0.2)", bg: "#0f111a" }
      : { dot: "#311B92", link: "rgba(33,150,243,0.15)", bg: "#fafafa" };
  };

  let colors = getColor();

  // Handle window resize
  const resizeCanvas = () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  };
  window.addEventListener("resize", resizeCanvas);
  resizeCanvas();

  // Initialize particles
  for (let i = 0; i < particleCount; i++) {
    particles.push({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * maxSpeed,
      vy: (Math.random() - 0.5) * maxSpeed,
      radius: Math.random() * 1.6 + 0.6
    });
  }

  // Animation loop
  const draw = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    colors = getColor();

    // Fade background (for subtle motion trails)
    ctx.fillStyle = colors.bg;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw and move particles
    for (let i = 0; i < particleCount; i++) {
      const p = particles[i];
      p.x += p.vx;
      p.y += p.vy;

      // Bounce off edges
      if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
      if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

      // Draw dot
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius, 0, 2 * Math.PI);
      ctx.fillStyle = colors.dot;
      ctx.fill();
    }

    // Draw connections
    for (let i = 0; i < particleCount; i++) {
      for (let j = i + 1; j < particleCount; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < linkDistance) {
          ctx.beginPath();
          ctx.strokeStyle = colors.link;
          ctx.lineWidth = 0.8;
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
        }
      }
    }

    requestAnimationFrame(draw);
  };

  draw();
});
