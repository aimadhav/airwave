/* Public static hosts may ignore Range requests. Buffer once, then seek locally. */
(() => {
  'use strict';
  /**
   * @param {{audio: HTMLAudioElement, source: string, mimeType: string,
   * timeoutMs: number, onStatus: (message: string) => void}} options
   * @returns {{ensure: () => Promise<void>}}
   */
  window.createBufferedPlayback = ({audio, source, mimeType, timeoutMs, onStatus}) => {
    let ready = false, pending = null, objectUrl = null;
    /** @returns {Promise<void>} */
    async function load() {
      const controller = new AbortController();
      let timer;
      onStatus('Loading web audio for reliable timestamp seeking…');
      try {
        await Promise.race([
          (async () => {
            const response = await fetch(source, {signal: controller.signal, credentials: 'omit'});
            if (!response.ok) throw new Error('Audio download failed');
            const bytes = await response.blob();
            objectUrl = URL.createObjectURL(new Blob([bytes], {type: mimeType}));
            await new Promise((resolve, reject) => {
              const cleanup = () => {
                audio.removeEventListener('loadedmetadata', loaded);
                audio.removeEventListener('error', failed);
                controller.signal.removeEventListener('abort', failed);
              };
              const loaded = () => {cleanup(); resolve();};
              const failed = () => {cleanup(); reject(new Error('Audio could not be decoded'));};
              audio.addEventListener('loadedmetadata', loaded, {once: true});
              audio.addEventListener('error', failed, {once: true});
              controller.signal.addEventListener('abort', failed, {once: true});
              audio.src = objectUrl;
              audio.load();
            });
          })(),
          new Promise((_, reject) => {timer = setTimeout(() => {
            controller.abort(); reject(new Error('Audio download timed out'));
          }, timeoutMs);}),
        ]);
        ready = true;
        onStatus('Audio ready. Timestamp seeking is available.');
      } catch (error) {
        if (objectUrl) {URL.revokeObjectURL(objectUrl); objectUrl = null;}
        onStatus('Web audio could not load. Check your connection and click again to retry.');
        throw error;
      } finally {clearTimeout(timer);}
    }
    return {
      /** @returns {Promise<void>} */
      ensure() {
        if (ready) return Promise.resolve();
        if (!pending) pending = load().catch(error => {pending = null; throw error;});
        return pending;
      },
    };
  };
})();
