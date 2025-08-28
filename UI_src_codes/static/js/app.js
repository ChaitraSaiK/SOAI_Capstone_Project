/**
 * Medical Assistant Chatbot Frontend
 * Interactive chat interface for medical AI assistant
 */

class MedicalChatbot {
    constructor() {
        this.sessionId = null;
        this.isLoading = false;
        this.currentImageFile = null;
        
        // DOM elements
        this.messagesContainer = document.getElementById('messagesContainer');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendMessage');
        this.attachButton = document.getElementById('attachImage');
        this.clearButton = document.getElementById('clearChat');
        this.newSessionButton = document.getElementById('newSession');
        this.sessionIdSpan = document.getElementById('sessionId');
        this.loadingIndicator = document.getElementById('loadingIndicator');
        
        // Image upload elements
        this.imageUploadContainer = document.getElementById('imageUploadContainer');
        this.textInputContainer = document.getElementById('textInputContainer');
        this.uploadArea = document.getElementById('uploadArea');
        this.imageInput = document.getElementById('imageInput');
        this.imagePreview = document.getElementById('imagePreview');
        this.previewImage = document.getElementById('previewImage');
        this.imageDescription = document.getElementById('imageDescription');
        this.uploadButton = document.getElementById('uploadImage');
        this.cancelUploadButton = document.getElementById('cancelUpload');
        this.removeImageButton = document.getElementById('removeImage');
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.generateSessionId();
        this.setupDragAndDrop();
        this.setupClipboardPaste();
        this.adjustTextareaHeight();
    }
    
    setupEventListeners() {
        // Text input events
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        this.messageInput.addEventListener('input', () => this.adjustTextareaHeight());
        
        // Image upload events
        this.attachButton.addEventListener('click', () => this.showImageUpload());
        this.uploadArea.addEventListener('click', () => this.imageInput.click());
        this.imageInput.addEventListener('change', (e) => this.handleImageSelect(e));
        this.uploadButton.addEventListener('click', () => this.uploadImage());
        this.cancelUploadButton.addEventListener('click', () => this.hideImageUpload());
        this.removeImageButton.addEventListener('click', () => this.removeImage());
        
        // Session management
        this.clearButton.addEventListener('click', () => this.clearChat());
        this.newSessionButton.addEventListener('click', () => this.startNewSession());
    }
    
    setupDragAndDrop() {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            this.uploadArea.addEventListener(eventName, this.preventDefaults, false);
        });
        
        ['dragenter', 'dragover'].forEach(eventName => {
            this.uploadArea.addEventListener(eventName, () => {
                this.uploadArea.parentElement.classList.add('dragover');
            }, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            this.uploadArea.addEventListener(eventName, () => {
                this.uploadArea.parentElement.classList.remove('dragover');
            }, false);
        });
        
        this.uploadArea.addEventListener('drop', (e) => this.handleDrop(e), false);
    }
    
    setupClipboardPaste() {
        // Listen for paste events on the document
        document.addEventListener('paste', (e) => this.handleClipboardPaste(e));
        
        // Also listen specifically on message input
        this.messageInput.addEventListener('paste', (e) => this.handleClipboardPaste(e));
    }
    
    async handleClipboardPaste(e) {
        const items = e.clipboardData?.items;
        if (!items) return;
        
        // Look for image in clipboard
        for (let i = 0; i < items.length; i++) {
            const item = items[i];
            if (item.type.indexOf('image') !== -1) {
                e.preventDefault();
                
                const file = item.getAsFile();
                if (file) {
                    // Convert to base64 and send to clipboard endpoint
                    const reader = new FileReader();
                    reader.onload = async (event) => {
                        const base64Data = event.target.result;
                        await this.handleClipboardImage(base64Data);
                    };
                    reader.readAsDataURL(file);
                }
                break;
            }
        }
    }
    
    async handleClipboardImage(base64Data) {
        if (this.isLoading) return;
        
        this.isLoading = true;
        this.showLoading();
        
        // Show paste notification
        this.showToast('Image pasted from clipboard! Analyzing...', 'info');
        
        // Add user message with pasted image
        this.addImageMessageFromBase64(base64Data, "", 'user');
        
        try {
            const formData = new FormData();
            formData.append('image_data', base64Data);
            formData.append('description', '');
            formData.append('session_id', this.sessionId);
            
            const response = await fetch('/api/chat/image/clipboard', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            this.sessionId = data.session_id;
            this.sessionIdSpan.textContent = this.sessionId;
            
            // Add assistant response
            this.addMessage(data.response, 'assistant');
            
        } catch (error) {
            console.error('Error analyzing clipboard image:', error);
            this.showToast('Failed to analyze clipboard image. Please try again.', 'error');
            this.addMessage('Sorry, I encountered an error analyzing the clipboard image. Please try again.', 'assistant');
        } finally {
            this.isLoading = false;
            this.hideLoading();
        }
    }
    
    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    handleDrop(e) {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.handleImageFile(files[0]);
        }
    }
    
    adjustTextareaHeight() {
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
    }
    
    generateSessionId() {
        this.sessionId = 'session_' + Math.random().toString(36).substr(2, 9);
        this.sessionIdSpan.textContent = this.sessionId;
    }
    
    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isLoading) return;
        
        this.isLoading = true;
        this.showLoading();
        
        // Add user message to chat
        this.addMessage(message, 'user');
        this.messageInput.value = '';
        this.adjustTextareaHeight();
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    session_id: this.sessionId
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            this.sessionId = data.session_id;
            this.sessionIdSpan.textContent = this.sessionId;
            
            // Add assistant response
            this.addMessage(data.response, 'assistant');
            
        } catch (error) {
            console.error('Error sending message:', error);
            this.showToast('Failed to send message. Please try again.', 'error');
            this.addMessage('Sorry, I encountered an error. Please try again.', 'assistant');
        } finally {
            this.isLoading = false;
            this.hideLoading();
        }
    }
    
    showImageUpload() {
        this.textInputContainer.style.display = 'none';
        this.imageUploadContainer.style.display = 'block';
    }
    
    hideImageUpload() {
        this.imageUploadContainer.style.display = 'none';
        this.textInputContainer.style.display = 'block';
        this.resetImageUpload();
    }
    
    handleImageSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.handleImageFile(file);
        }
    }
    
    handleImageFile(file) {
        if (!this.validateImage(file)) {
            this.showToast('Please select a valid image file (JPG, PNG, GIF) under 10MB.', 'error');
            return;
        }
        
        this.currentImageFile = file;
        this.showImagePreview(file);
        this.uploadButton.disabled = false;
    }
    
    validateImage(file) {
        const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp'];
        const maxSize = 10 * 1024 * 1024; // 10MB
        
        return allowedTypes.includes(file.type) && file.size <= maxSize;
    }
    
    showImagePreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            this.previewImage.src = e.target.result;
            this.imagePreview.style.display = 'block';
            this.uploadArea.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }
    
    removeImage() {
        this.resetImageUpload();
        this.uploadButton.disabled = true;
    }
    
    resetImageUpload() {
        this.currentImageFile = null;
        this.imageInput.value = '';
        this.imageDescription.value = '';
        this.imagePreview.style.display = 'none';
        this.uploadArea.style.display = 'block';
        this.previewImage.src = '';
    }
    
    async uploadImage() {
        if (!this.currentImageFile || this.isLoading) return;
        
        this.isLoading = true;
        this.showLoading();
        
        const description = this.imageDescription.value.trim();
        
        // Add user message with image
        this.addImageMessage(this.currentImageFile, description, 'user');
        
        try {
            const formData = new FormData();
            formData.append('file', this.currentImageFile);
            formData.append('description', description);
            formData.append('session_id', this.sessionId);
            
            const response = await fetch('/api/chat/image', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            this.sessionId = data.session_id;
            this.sessionIdSpan.textContent = this.sessionId;
            
            // Add assistant response
            this.addMessage(data.response, 'assistant');
            
            // Hide image upload and reset
            this.hideImageUpload();
            
        } catch (error) {
            console.error('Error uploading image:', error);
            this.showToast('Failed to analyze image. Please try again.', 'error');
            this.addMessage('Sorry, I encountered an error analyzing the image. Please try again.', 'assistant');
        } finally {
            this.isLoading = false;
            this.hideLoading();
        }
    }
    
    addMessage(content, sender, imageUrl = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        if (imageUrl) {
            const img = document.createElement('img');
            img.src = imageUrl;
            img.className = 'message-image';
            img.alt = 'Uploaded image';
            contentDiv.appendChild(img);
        }
        
        // Only add text div if there's content
        if (content && content.trim()) {
            const textDiv = document.createElement('div');
            
            // Format medical responses for assistant messages
            if (sender === 'assistant') {
                textDiv.innerHTML = this.formatMedicalResponse(content);
                textDiv.className = 'medical-report';
            } else {
                textDiv.textContent = content;
            }
            
            contentDiv.appendChild(textDiv);
        }
        
        const timestampDiv = document.createElement('div');
        timestampDiv.className = 'message-timestamp';
        timestampDiv.textContent = new Date().toLocaleTimeString();
        contentDiv.appendChild(timestampDiv);
        
        messageDiv.appendChild(contentDiv);
        
        // Remove welcome message if it exists
        const welcomeMessage = this.messagesContainer.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.remove();
        }
        
        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    addImageMessage(file, description, sender) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const message = description || `Uploaded image: ${file.name}`;
            this.addMessage(message, sender, e.target.result);
        };
        reader.readAsDataURL(file);
    }
    
    addImageMessageFromBase64(base64Data, description, sender) {
        // Only show message if there's a description, otherwise just show the image
        const message = description || "";
        this.addMessage(message, sender, base64Data);
    }
    
    formatMedicalResponse(text) {
        // Escape HTML first
        let formattedText = text.replace(/</g, '&lt;').replace(/>/g, '&gt;');
        
        // Format section headers (text followed by colon and optional asterisks)
        formattedText = formattedText.replace(/\*?\*?([A-Z][^:*\n]*?):\*?\*?/g, '<h3>$1</h3>');
        
        // Format subsection headers (bold text with asterisks)
        formattedText = formattedText.replace(/\* \*\*([^*]+)\*\*:/g, '<h4>$1</h4>');
        
        // Format bullet points that start with *
        formattedText = formattedText.replace(/^\* ([^*\n]+)$/gm, '<li>$1</li>');
        
        // Wrap consecutive list items in ul tags
        formattedText = formattedText.replace(/(<li>.*?<\/li>(\s*<li>.*?<\/li>)*)/gs, '<ul>$1</ul>');
        
        // Format bold text within content
        formattedText = formattedText.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        
        // Format specific medical sections
        if (formattedText.includes('Conclusion')) {
            formattedText = formattedText.replace(
                /<h3>Conclusion<\/h3>/g, 
                '<div class="conclusion"><h4>Conclusion</h4>'
            );
            
            // Find the next h3 or end of text and close the conclusion div
            formattedText = formattedText.replace(
                /(<div class="conclusion">.*?)(<h3>|$)/gs, 
                '$1</div>$2'
            );
        }
        
        if (formattedText.includes('Disclaimer')) {
            formattedText = formattedText.replace(
                /<h3>Disclaimer<\/h3>/g, 
                '<div class="disclaimer"><strong>Disclaimer</strong>'
            );
            
            // Find the next h3 or end of text and close the disclaimer div
            formattedText = formattedText.replace(
                /(<div class="disclaimer">.*?)(<h3>|$)/gs, 
                '$1</div>$2'
            );
        }
        
        // Convert line breaks to paragraphs for better spacing
        formattedText = formattedText.replace(/\n\n+/g, '</p><p>');
        formattedText = formattedText.replace(/\n/g, '<br>');
        
        // Wrap in paragraph tags if not already wrapped
        if (!formattedText.startsWith('<')) {
            formattedText = '<p>' + formattedText + '</p>';
        }
        
        // Clean up empty paragraphs
        formattedText = formattedText.replace(/<p><\/p>/g, '');
        formattedText = formattedText.replace(/<p><br><\/p>/g, '');
        
        return formattedText;
    }
    
    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }
    
    showLoading() {
        this.loadingIndicator.style.display = 'flex';
        this.sendButton.disabled = true;
        this.uploadButton.disabled = true;
    }
    
    hideLoading() {
        this.loadingIndicator.style.display = 'none';
        this.sendButton.disabled = false;
        if (this.currentImageFile) {
            this.uploadButton.disabled = false;
        }
    }
    
    clearChat() {
        if (confirm('Are you sure you want to clear the chat history?')) {
            // Remove all messages except welcome message
            const messages = this.messagesContainer.querySelectorAll('.message');
            messages.forEach(message => message.remove());
            
            // Add welcome message back
            this.addWelcomeMessage();
        }
    }
    
    async startNewSession() {
        if (confirm('Start a new session? This will clear the current chat.')) {
            try {
                // End current session
                if (this.sessionId) {
                    await fetch(`/api/chat/session/${this.sessionId}`, {
                        method: 'DELETE'
                    });
                }
                
                // Generate new session
                this.generateSessionId();
                
                // Clear chat
                const messages = this.messagesContainer.querySelectorAll('.message');
                messages.forEach(message => message.remove());
                this.addWelcomeMessage();
                
                this.showToast('New session started!', 'success');
                
            } catch (error) {
                console.error('Error starting new session:', error);
                this.showToast('Failed to start new session.', 'error');
            }
        }
    }
    
    addWelcomeMessage() {
        const welcomeHtml = `
            <div class="welcome-message">
                <div class="welcome-icon">
                    <i class="fas fa-stethoscope"></i>
                </div>
                <h2>Welcome to Medical Assistant</h2>
                <p>I'm here to help with medical questions and image analysis. You can:</p>
                <ul>
                    <li><i class="fas fa-comment"></i> Ask medical questions</li>
                    <li><i class="fas fa-image"></i> Upload medical images for analysis</li>
                    <li><i class="fas fa-search"></i> Get detailed explanations</li>
                </ul>
                <p class="disclaimer">
                    <i class="fas fa-exclamation-triangle"></i>
                    <strong>Disclaimer:</strong> This AI assistant is for educational purposes only. 
                    Always consult with qualified healthcare professionals for medical advice.
                </p>
            </div>
        `;
        this.messagesContainer.innerHTML = welcomeHtml;
    }
    
    showToast(message, type = 'info') {
        const toastContainer = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        
        toastContainer.appendChild(toast);
        
        // Auto remove after 3 seconds
        setTimeout(() => {
            toast.remove();
        }, 3000);
    }
}

// Initialize the chatbot when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const chatbot = new MedicalChatbot();
    
    // Global error handler
    window.addEventListener('error', (e) => {
        console.error('Global error:', e.error);
        chatbot.showToast('An unexpected error occurred.', 'error');
    });
    
    // Handle offline/online status
    window.addEventListener('offline', () => {
        chatbot.showToast('Connection lost. Please check your internet.', 'warning');
    });
    
    window.addEventListener('online', () => {
        chatbot.showToast('Connection restored!', 'success');
    });
});
