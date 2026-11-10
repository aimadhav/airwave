(() => {
  const nodes = [...document.querySelectorAll('.journey-node')];
  const sections = [...document.querySelectorAll('.guide-section[id]')];
  function setActive(id) {
    nodes.forEach((node) => node.classList.toggle('current', node.dataset.go === id));
  }

  nodes.forEach((node) => {
    const go = () => {
      const destination = document.getElementById(node.dataset.go);
      if (!destination) return;
      setActive(node.dataset.go);
      destination.scrollIntoView({ behavior: 'smooth', block: 'start' });
    };
    node.addEventListener('click', go);
    node.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); go(); }
    });
  });

  const observer = new IntersectionObserver((entries) => {
    const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (visible) setActive(visible.target.id);
  }, { rootMargin: '-18% 0px -62% 0px', threshold: [0.1, 0.35, 0.7] });
  sections.forEach((section) => observer.observe(section));

})();
