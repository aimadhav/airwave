(() => {
  const landing = document.querySelector('#landing-page');
  const studio = document.querySelector('#studio-app');
  if (!landing || !studio) return;

  function enterStudio({ push = true } = {}) {
    landing.hidden = true;
    studio.hidden = false;
    document.body.classList.add('studio-open');
    document.title = 'Airwave Studio — Moment-aware podcast ads';
    if (push && location.hash !== '#studio') history.pushState({ view: 'studio' }, '', '#studio');
    window.scrollTo({ top: 0, behavior: 'auto' });
    requestAnimationFrame(() => window.dispatchEvent(new Event('resize')));
  }

  function enterLanding({ push = true } = {}) {
    studio.hidden = true;
    landing.hidden = false;
    document.body.classList.remove('studio-open');
    document.title = 'Airwave — Ads that flow with the conversation';
    if (push && location.hash !== '#landing-home') history.pushState({ view: 'landing' }, '', '#landing-home');
    window.scrollTo({ top: 0, behavior: 'auto' });
  }

  document.querySelectorAll('[data-enter-studio]').forEach(button => button.addEventListener('click', () => enterStudio()));
  document.querySelectorAll('[data-show-landing]').forEach(link => link.addEventListener('click', event => {
    event.preventDefault();
    enterLanding();
  }));

  function animateDemo(target) {
    const demo = document.querySelector('#landing-demo');
    if (!demo) return;
    demo.scrollIntoView({ behavior: 'smooth', block: 'center' });
    demo.classList.add('is-playing');
    demo.querySelectorAll('.compare-card').forEach(card => {
      card.classList.add('is-playing');
      const button = card.querySelector('.compare-play');
      if (button) {
        button.textContent = 'Ⅱ';
        button.setAttribute('aria-label', 'Pause visual demo');
      }
    });
    window.clearTimeout(animateDemo.timer);
    animateDemo.timer = window.setTimeout(() => {
      demo.classList.remove('is-playing');
      demo.querySelectorAll('.compare-card').forEach(card => {
        card.classList.remove('is-playing');
        const button = card.querySelector('.compare-play');
        if (button) {
          button.textContent = '▶';
          button.setAttribute('aria-label', card.classList.contains('compare-airflow') ? 'Animate Airwave demo' : 'Animate usual ad break');
        }
      });
    }, 8000);
  }

  document.querySelectorAll('[data-show-demo]').forEach(link => link.addEventListener('click', event => {
    event.preventDefault();
    animateDemo(link);
  }));

  document.querySelectorAll('.compare-play').forEach(button => button.addEventListener('click', () => {
    const card = button.closest('.compare-card');
    const playing = card.classList.toggle('is-playing');
    button.textContent = playing ? 'Ⅱ' : '▶';
    button.setAttribute('aria-label', playing ? 'Pause visual demo' : 'Play visual demo');
  }));

  const voiceCard = document.querySelector('.voice-demo-card');
  const voicePlay = document.querySelector('.voice-play');
  voicePlay?.addEventListener('click', () => {
    const playing = voiceCard.classList.toggle('is-playing');
    voicePlay.textContent = playing ? 'Ⅱ' : '▶';
    voicePlay.setAttribute('aria-label', playing ? 'Pause voice preview animation' : 'Play voice preview animation');
  });

  document.querySelectorAll('.voice-option').forEach(button => button.addEventListener('click', () => {
    document.querySelectorAll('.voice-option').forEach(option => {
      const selected = option === button;
      option.classList.toggle('active', selected);
      option.setAttribute('aria-pressed', String(selected));
    });
    voiceCard?.classList.add('voice-changed');
    window.setTimeout(() => voiceCard?.classList.remove('voice-changed'), 420);
  }));

  const revealItems = document.querySelectorAll('.flow-step-card,.use-card,.compare-card,.benefits-row>div,.landing-cta-banner');
  if ('IntersectionObserver' in window) {
    landing.classList.add('reveal-ready');
    const revealObserver = new IntersectionObserver(entries => entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        revealObserver.unobserve(entry.target);
      }
    }), { threshold: .14 });
    revealItems.forEach((element, index) => {
      element.classList.add('reveal-item');
      element.style.setProperty('--reveal-delay', `${Math.min(index % 5, 4) * 65}ms`);
      revealObserver.observe(element);
    });
  }

  window.addEventListener('popstate', () => {
    if (location.hash === '#studio') enterStudio({ push: false });
    else if (!studio.hidden) enterLanding({ push: false });
  });

  if (location.hash === '#studio') enterStudio({ push: false });
})();

