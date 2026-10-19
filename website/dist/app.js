'use strict';

const menu = document.querySelector('.menu-toggle');
const navigation = document.querySelector('#navigation');
menu.addEventListener('click', () => {
  const expanded = menu.getAttribute('aria-expanded') !== 'true';
  menu.setAttribute('aria-expanded', String(expanded));
  menu.setAttribute('aria-label', expanded ? 'Close navigation' : 'Open navigation');
  navigation.classList.toggle('open', expanded);
});
navigation.querySelectorAll('a').forEach(link => link.addEventListener('click', () => {
  menu.setAttribute('aria-expanded', 'false');
  menu.setAttribute('aria-label', 'Open navigation');
  navigation.classList.remove('open');
}));

const teaser = document.querySelector('#teaser-video');
const teaserToggle = document.querySelector('#teaser-toggle');
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
function updatePreviewButton() {
  teaserToggle.innerHTML = teaser.paused ? 'Play preview <span aria-hidden="true">▷</span>' : 'Pause preview <span aria-hidden="true">Ⅱ</span>';
  teaserToggle.setAttribute('aria-label', teaser.paused ? 'Play preview' : 'Pause preview');
  teaserToggle.setAttribute('aria-pressed', String(teaser.paused));
}
if (reducedMotion.matches) {
  teaser.autoplay = false;
  teaser.pause();
}
teaser.addEventListener('play', updatePreviewButton);
teaser.addEventListener('pause', updatePreviewButton);
teaserToggle.addEventListener('click', () => {
  if (teaser.paused) teaser.play().catch(updatePreviewButton);
  else teaser.pause();
});
updatePreviewButton();

const overview = document.querySelector('#overview-video');
document.querySelectorAll('[data-video-time]').forEach(button => button.addEventListener('click', () => {
  const time = Number(button.dataset.videoTime);
  document.querySelectorAll('[data-video-time]').forEach(item => item.classList.toggle('active', item === button));
  if (overview.readyState === 0) {
    overview.addEventListener('loadedmetadata', () => { overview.currentTime = time; }, { once: true });
    overview.load();
  } else overview.currentTime = time;
  overview.play().catch(() => overview.focus());
}));

const dialog = document.querySelector('#figure-dialog');
const image = document.querySelector('#dialog-image');
document.querySelectorAll('[data-zoom]').forEach(button => button.addEventListener('click', () => {
  image.src = button.dataset.zoom;
  image.alt = button.querySelector('img').alt;
  document.querySelector('#dialog-caption').textContent = button.dataset.caption;
  document.querySelector('#dialog-original').href = button.dataset.zoom;
  dialog.showModal();
}));
document.querySelector('.dialog-close').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });

async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
  const field = document.createElement('textarea');
  field.value = text;
  field.style.position = 'fixed';
  field.style.opacity = '0';
  document.body.appendChild(field);
  field.select();
  const copied = document.execCommand('copy');
  field.remove();
  if (!copied) throw new Error('Copy unavailable');
}
document.querySelectorAll('[data-copy-target]').forEach(button => button.addEventListener('click', async () => {
  const label = button.querySelector('span');
  const original = label.textContent;
  try {
    await copyText(document.getElementById(button.dataset.copyTarget).textContent.trim());
    label.textContent = 'Copied!';
    document.querySelector('#copy-status').textContent = button.dataset.copyTarget === 'bibtex' ? 'BibTeX citation copied to clipboard.' : 'Installation commands copied to clipboard.';
  } catch {
    label.textContent = 'Select text to copy';
    document.querySelector('#copy-status').textContent = 'Clipboard access is unavailable. Please select and copy the text.';
  }
  setTimeout(() => { label.textContent = original; }, 2500);
}));

if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        navigation.querySelectorAll('a[href^="#"]').forEach(link => link.classList.toggle('active', link.hash === `#${entry.target.id}`));
      }
    });
  }, { rootMargin: '-15% 0px -55% 0px', threshold: 0 });
  ['method', 'experiments', 'video', 'citation'].forEach(id => observer.observe(document.getElementById(id)));
}
