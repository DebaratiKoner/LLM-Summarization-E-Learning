function toggleInputSections() {
    const inputType = document.getElementById("inputType");
    const textSection = document.getElementById("textSection");
    const videoSection = document.getElementById("videoSection");
    const audioSection = document.getElementById("audioSection");
    const imageSection = document.getElementById("imageSection");

    if (!inputType) {
        return;
    }

    const selected = inputType.value;
    textSection.style.display = selected === "Text" ? "block" : "none";
    videoSection.style.display = selected === "Video" ? "block" : "none";
    audioSection.style.display = selected === "Audio" ? "block" : "none";
    imageSection.style.display = selected === "Image" ? "block" : "none";
}

function toggleTextOptions() {
    const selected = document.querySelector('input[name="text_option"]:checked');
    const typedTextWrap = document.getElementById("typedTextWrap");
    const pdfWrap = document.getElementById("pdfWrap");
    const docxWrap = document.getElementById("docxWrap");

    if (!selected) {
        return;
    }

    typedTextWrap.style.display = selected.value === "Type Text" ? "block" : "none";
    pdfWrap.style.display = selected.value === "Upload PDF" ? "block" : "none";
    docxWrap.style.display = selected.value === "Upload DOCX" ? "block" : "none";
}

document.addEventListener("DOMContentLoaded", () => {
    const inputType = document.getElementById("inputType");
    const textOptions = document.querySelectorAll('input[name="text_option"]');
    const body = document.body;
    const navLinks = document.querySelectorAll(".site-nav a");
    const pageSections = document.querySelectorAll(".page-section");
    const chatWidgetToggle = document.getElementById("chatWidgetToggle");
    const chatWidgetClose = document.getElementById("chatWidgetClose");
    const chatWidgetCloseSecondary = document.getElementById("chatWidgetCloseSecondary");
    const chatWidgetPanel = document.getElementById("chatWidgetPanel");
    const notesWorkspace = document.querySelector(".notes-workspace");
    const podcastControls = document.querySelector(".podcast-controls");
    const restoreView = () => {
        if (!body) {
            return;
        }

        const viewSection = body.dataset.viewSection;
        let targetId = "";

        if (viewSection === "analysis") {
            targetId = body.dataset.hasOutput === "true" ? "output" : "analysis";
        } else if (viewSection === "contact") {
            targetId = "contact";
        } else if (viewSection === "home") {
            targetId = "home";
        }

        if (!targetId) {
            return;
        }

        const target = document.getElementById(targetId);
        if (target) {
            target.scrollIntoView({ behavior: "auto", block: "start" });
        }
    };

    if (inputType) {
        inputType.addEventListener("change", toggleInputSections);
    }

    textOptions.forEach((option) => {
        option.addEventListener("change", toggleTextOptions);
    });

    toggleInputSections();
    toggleTextOptions();

    if (navLinks.length && pageSections.length) {
        const updateActiveLink = (id) => {
            navLinks.forEach((link) => {
                link.classList.toggle("active", link.getAttribute("href") === `#${id}`);
            });
        };

        const observer = new IntersectionObserver(
            (entries) => {
                const visibleEntry = entries
                    .filter((entry) => entry.isIntersecting)
                    .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

                if (visibleEntry) {
                    updateActiveLink(visibleEntry.target.id);
                }
            },
            {
                threshold: 0.35,
                rootMargin: "-10% 0px -35% 0px",
            }
        );

        pageSections.forEach((section) => observer.observe(section));
        updateActiveLink(body && body.dataset.viewSection ? body.dataset.viewSection : "home");
    }

    if (chatWidgetToggle && chatWidgetPanel) {
        const setChatWidgetState = (isOpen) => {
            chatWidgetPanel.hidden = !isOpen;
            chatWidgetToggle.setAttribute("aria-expanded", String(isOpen));
        };

        chatWidgetToggle.addEventListener("click", () => {
            setChatWidgetState(chatWidgetPanel.hidden);
        });

        if (chatWidgetClose) {
            chatWidgetClose.addEventListener("click", () => {
                setChatWidgetState(false);
            });
        }

        if (chatWidgetCloseSecondary) {
            chatWidgetCloseSecondary.addEventListener("click", () => {
                setChatWidgetState(false);
            });
        }

        document.addEventListener("keydown", (event) => {
            if (event.key === "Escape") {
                setChatWidgetState(false);
            }
        });

        setChatWidgetState(body && body.dataset.openChatWidget === "true");
    }

    if (podcastControls && "speechSynthesis" in window) {
        const text = podcastControls.dataset.audioText || "";
        const playButton = podcastControls.querySelector(".podcast-play");
        const stopButton = podcastControls.querySelector(".podcast-stop");

        if (playButton) {
            playButton.addEventListener("click", () => {
                window.speechSynthesis.cancel();
                if (!text.trim()) {
                    return;
                }
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.rate = 0.95;
                utterance.pitch = 1;
                window.speechSynthesis.speak(utterance);
            });
        }

        if (stopButton) {
            stopButton.addEventListener("click", () => {
                window.speechSynthesis.cancel();
            });
        }
    }

    if (notesWorkspace) {
        const topic = notesWorkspace.dataset.topic || "study-topic";
        const textarea = notesWorkspace.querySelector(".notes-textarea");
        const saveButton = notesWorkspace.querySelector(".note-save");
        const recordButton = notesWorkspace.querySelector(".note-record");
        const stopButton = notesWorkspace.querySelector(".note-stop");
        const status = notesWorkspace.querySelector(".notes-status");
        const audioPreview = notesWorkspace.querySelector(".notes-audio-preview");
        const storageKey = `apsg-notes-${topic.toLowerCase()}`;
        const audioKey = `apsg-notes-audio-${topic.toLowerCase()}`;
        let mediaRecorder = null;
        let audioChunks = [];

        if (textarea) {
            const savedNotes = window.localStorage.getItem(storageKey);
            if (savedNotes) {
                textarea.value = savedNotes;
            }
        }

        if (audioPreview) {
            const savedAudio = window.localStorage.getItem(audioKey);
            if (savedAudio) {
                audioPreview.src = savedAudio;
                audioPreview.hidden = false;
            }
        }

        if (saveButton && textarea) {
            saveButton.addEventListener("click", () => {
                window.localStorage.setItem(storageKey, textarea.value);
                if (status) {
                    status.textContent = "Notes saved for this topic in your browser.";
                }
            });
        }

        if (recordButton && stopButton && navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            recordButton.addEventListener("click", async () => {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];
                    mediaRecorder.ondataavailable = (event) => {
                        if (event.data.size > 0) {
                            audioChunks.push(event.data);
                        }
                    };
                    mediaRecorder.onstop = () => {
                        const blob = new Blob(audioChunks, { type: "audio/webm" });
                        const objectUrl = URL.createObjectURL(blob);
                        if (audioPreview) {
                            audioPreview.src = objectUrl;
                            audioPreview.hidden = false;
                        }
                        const reader = new FileReader();
                        reader.onloadend = () => {
                            window.localStorage.setItem(audioKey, reader.result);
                            if (status) {
                                status.textContent = "Voice note saved for this topic in your browser.";
                            }
                        };
                        reader.readAsDataURL(blob);
                        stream.getTracks().forEach((track) => track.stop());
                    };
                    mediaRecorder.start();
                    recordButton.disabled = true;
                    stopButton.disabled = false;
                    if (status) {
                        status.textContent = "Recording voice note...";
                    }
                } catch (error) {
                    if (status) {
                        status.textContent = "Microphone access was not available for recording.";
                    }
                }
            });

            stopButton.addEventListener("click", () => {
                if (mediaRecorder && mediaRecorder.state !== "inactive") {
                    mediaRecorder.stop();
                }
                recordButton.disabled = false;
                stopButton.disabled = true;
            });
        }
    }

    restoreView();
});
