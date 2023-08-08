const chatHistory = document.getElementById('chat-history');
const inputMsg = document.getElementById('input-msg');
const sendBtn = document.getElementById('send-btn');

sendBtn.addEventListener('click', sendMessage);

function sendMessage() {
    const userMessage = inputMsg.value;
    if (userMessage.trim() === '') return;

    addMessageToHistory(userMessage, 'user');
    inputMsg.value = '';

    fetchResponseFromAPI(userMessage);
}

function addMessageToHistory(message, sender) {
    const messageElement = document.createElement('div');
    messageElement.className = `message ${sender}`;
    messageElement.textContent = message;

    chatHistory.appendChild(messageElement);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function fetchResponseFromAPI(userMessage) {
    // Replace 'YOUR_API_KEY' with your actual OpenAI GPT-3 API key
    const apiKey = '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';
    
    fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify({
            model: 'gpt-3.5-turbo',
            messages: [{ role: 'system', content: 'You are a helpful assistant.' }, { role: 'user', content: userMessage }]
        })
    })
    .then(response => response.json())
    .then(data => {
        const aiResponse = data.choices[0].message.content;
        addMessageToHistory(aiResponse, 'ai');
    })
    .catch(error => {
        console.error('Error fetching API response:', error);
    });
}



//  FABRIC JS


let canvas = null;
const addedObjects = [];

// Function to handle file upload
function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();

  // Listen for the file to be loaded
  reader.onload = function (e) {
    try {
      const jsonData = JSON.parse(e.target.result);
      initializeCanvas(jsonData);
    } catch (error) {
      console.error('Error parsing JSON file:', error);
      alert('Error parsing JSON file. Please make sure the file is valid JSON.');
    }
  };

  // Read the file as text
  reader.readAsText(file);
}

// Listen for changes in the file input element
document.getElementById('jsonFileInput').addEventListener('change', handleFileUpload);

const loadedFonts = {}; // Font cache

function loadCustomFont(fontFamily, fontUrl) {
  if (!fontFamily || !fontUrl) {
    return Promise.reject(new Error('Font family or URL is missing.'));
  }

  if (loadedFonts[fontFamily]) {
    return loadedFonts[fontFamily];
  }

  const font = new FontFace(fontFamily, `url(${fontUrl})`);
  loadedFonts[fontFamily] = font.load();
  return loadedFonts[fontFamily];
}

function initializeCanvas(data) {
  const canvasWidth = 800;
  const canvasHeight = 600;

  if (canvas) {
    canvas.clear();
    addedObjects.length = 0;
  } else {
    canvas = new fabric.Canvas('canvas', {
      width: canvasWidth,
      height: canvasHeight
    });
  }

  let maxWidth = 0;
  let maxHeight = 0;
  let backgroundImage;

  const objectPromises = data.objects.map(objData => createFabricObject(objData));

  Promise.all(objectPromises).then(objects => {
    objects.forEach(obj => {
      canvas.add(obj);
      const objWidth = obj.left + obj.width * obj.scaleX;
      const objHeight = obj.top + obj.height * obj.scaleY;
      maxWidth = Math.max(maxWidth, objWidth);
      maxHeight = Math.max(maxHeight, objHeight);
    });

    const scaleFactor = Math.min(
      canvasWidth / maxWidth,
      canvasHeight / maxHeight
    );

    canvas.setZoom(scaleFactor);

    if (data.backgroundImage && data.backgroundImage.src) {
      fabric.Image.fromURL(data.backgroundImage.src, img => {
        backgroundImage = img;

        backgroundImage.set({
          left: data.backgroundImage.left || 0,
          top: data.backgroundImage.top || 0,
          width: data.backgroundImage.width || canvasWidth,
          height: data.backgroundImage.height || canvasHeight,
          selectable: false,
          evented: false,
        });

        canvas.add(backgroundImage);
        canvas.sendToBack(backgroundImage);
        canvas.renderAll();
      });
    }

    

    canvas.centerObject(backgroundImage);
    canvas.renderAll();
  });
}

function updateCanvasSize() {
  const scaleFactor = Math.min(
    canvas.getWidth() / canvas.width,
    canvas.getHeight() / canvas.height
  );

  canvas.setZoom(scaleFactor);
  canvas.renderAll();
}



function createFabricObject(objData, parentScaleFactor = 1) {
  switch (objData.type) {
    case 'image':
      return new Promise(resolve => {
        fabric.Image.fromURL(objData.src, img => {
          img.set(objData);
          resolve(img);
        });
      });case 'text':
      return new Promise(resolve => {
        if (objData.fontFamily && objData.fontUrl) {
          loadCustomFont(objData.fontFamily, objData.fontUrl)
            .then(() => {
              const text = new fabric.Text(objData.text, objData);
              resolve(text);
            })
            .catch(error => {
              console.error('Error loading font:', error);
              // Resolve with default text properties if font loading fails
              const text = new fabric.Text(objData.text, objData);
              resolve(text);
            });
        } else {
          // Resolve with default text properties if no font info provided
          const text = new fabric.Text(objData.text, objData);
          resolve(text);
        }
      });
    case 'group':
      return new Promise(resolve => {
        const groupObjects = objData.objects.map(groupObjData =>
          createFabricObject(groupObjData, objData.scaleX)
        );
        Promise.all(groupObjects).then(groupObjs => {
          const group = new fabric.Group(groupObjs, objData);
          group.set({ scaleX: objData.scaleX, scaleY: objData.scaleY });
          groupObjs.forEach((groupObj, index) => {
            groupObj.set({
              left: objData.objects[index].left,
              top: objData.objects[index].top,
            });
          });
          resolve(group);
        });
      });
    default:
      return null;
  }
}

