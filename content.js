chrome.webNavigation.onCompleted.addListener(function(details) {
  chrome.tabs.executeScript(
    { code: 'document.documentElement.outerHTML' },
    function(result) {
      if (chrome.runtime.lastError) {
        console.error(chrome.runtime.lastError.message);
      } else {
        chrome.runtime.sendMessage({ action: 'get_html', html: result[0] });
      }
    }
  );
});