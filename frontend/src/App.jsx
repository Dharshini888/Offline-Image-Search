// // import React, { useState, useEffect, useRef, useCallback } from "react";
// // import { motion, AnimatePresence } from "framer-motion";
// // import {
// //   Search, Upload, Image as ImgIcon, Clock, BookOpen, Users, Heart,
// //   Copy, Trash2, X, ChevronLeft, ChevronRight, Sparkles, BarChart3,
// //   GitMerge, Palette, Shuffle, RotateCcw, AlertTriangle, FileText,
// //   Mic, MicOff, Zap, Star, Eye, Camera, Info, Tag, Type, Smile,
// //   Loader2, Trash, BookImage, ImageOff, TrendingUp, Aperture,
// //   CheckSquare, Square, Plus, Calendar, Hash, FolderPlus,
// //   CheckCheck, Layers, ChevronDown, ChevronUp,
// //   SlidersHorizontal, UserCheck
// // } from "lucide-react";
// // import axios from "axios";

// // const API = "http://localhost:8000";
// // const imgUrl = (f) => {
// //   if (!f) return null;
// //   const b = String(f).split("/").filter(Boolean).pop();
// //   return b ? `${API}/image/${b}` : null;
// // };
// // const BLANK = `data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='1' height='1'%3E%3Crect fill='%23111118'/%3E%3C/svg%3E`;

// // const QL = {
// //   Excellent: { c:"#4ade80", bg:"rgba(74,222,128,.12)" },
// //   Good:      { c:"#818cf8", bg:"rgba(129,140,248,.12)" },
// //   Fair:      { c:"#fbbf24", bg:"rgba(251,191,36,.12)"  },
// //   Poor:      { c:"#f87171", bg:"rgba(248,113,113,.12)" },
// // };
// // const EMO = { happy:"😊",sad:"😢",angry:"😠",surprised:"😲",disgusted:"🤢",fearful:"😨",neutral:"😐" };
// // const EMO_COLORS = { happy:"#fbbf24",sad:"#60a5fa",angry:"#f87171",surprised:"#a78bfa",disgusted:"#4ade80",fearful:"#fb923c",neutral:"#6b7280" };
// // const qInfo = (l) => QL[l] || { c:"#555", bg:"rgba(255,255,255,.05)" };

// // /* ─── Voice hook ─────────────────────────────────────────────────────────── */
// // function useVoice(onResult, onError) {
// //   const [state, setState] = useState("idle");
// //   const streamRef = useRef(null);
// //   const processorRef = useRef(null);
// //   const ctxRef = useRef(null);
// //   const samplesRef = useRef([]);
// //   const ok = typeof navigator !== "undefined" && !!navigator.mediaDevices?.getUserMedia;

// //   // Convert Float32 PCM samples → 16-bit WAV blob (no ffmpeg needed)
// //   const buildWav = (samples, sampleRate) => {
// //     const len = samples.length;
// //     const buf = new ArrayBuffer(44 + len * 2);
// //     const view = new DataView(buf);
// //     const write = (off, str) => { for(let i=0;i<str.length;i++) view.setUint8(off+i, str.charCodeAt(i)); };
// //     write(0,"RIFF"); view.setUint32(4, 36+len*2, true);
// //     write(8,"WAVE"); write(12,"fmt ");
// //     view.setUint32(16,16,true); view.setUint16(20,1,true); view.setUint16(22,1,true);
// //     view.setUint32(24,sampleRate,true); view.setUint32(28,sampleRate*2,true);
// //     view.setUint16(32,2,true); view.setUint16(34,16,true);
// //     write(36,"data"); view.setUint32(40,len*2,true);
// //     for(let i=0;i<len;i++){
// //       const s = Math.max(-1,Math.min(1,samples[i]));
// //       view.setInt16(44+i*2, s<0?s*0x8000:s*0x7FFF, true);
// //     }
// //     return new Blob([buf], {type:"audio/wav"});
// //   };

// //   const start = useCallback(async () => {
// //     if (!ok) return onError?.("Mic unavailable");
// //     try {
// //       const stream = await navigator.mediaDevices.getUserMedia({
// //         audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true }
// //       });
// //       streamRef.current = stream;
// //       samplesRef.current = [];

// //       // Use Web Audio API to capture raw PCM at 16kHz — bypasses ffmpeg completely
// //       const ctx = new AudioContext({ sampleRate: 16000 });
// //       ctxRef.current = ctx;
// //       const source = ctx.createMediaStreamSource(stream);
// //       const processor = ctx.createScriptProcessor(4096, 1, 1);
// //       processorRef.current = processor;

// //       processor.onaudioprocess = e => {
// //         const data = e.inputBuffer.getChannelData(0);
// //         samplesRef.current.push(new Float32Array(data));
// //       };
// //       source.connect(processor);
// //       processor.connect(ctx.destination);
// //       setState("rec");
// //     } catch(e) {
// //       setState("error");
// //       onError?.(e.name === "NotAllowedError" ? "Mic permission denied" : "Mic unavailable");
// //       setTimeout(()=>setState("idle"),3000);
// //     }
// //   }, [ok]);

// //   const stop = useCallback(async () => {
// //     if (state !== "rec") return;
// //     setState("processing");
// //     try {
// //       // Stop all tracks
// //       streamRef.current?.getTracks().forEach(t=>t.stop());
// //       processorRef.current?.disconnect();
// //       await ctxRef.current?.close();

// //       // Flatten all captured chunks into one array
// //       const allSamples = samplesRef.current;
// //       if (!allSamples.length) {
// //         setState("error"); onError?.("No audio captured"); setTimeout(()=>setState("idle"),3000); return;
// //       }
// //       const total = allSamples.reduce((s,c)=>s+c.length,0);
// //       const merged = new Float32Array(total);
// //       let off=0; for(const c of allSamples){merged.set(c,off);off+=c.length;}

// //       // Build WAV and send to Vosk (no ffmpeg needed — already 16kHz mono PCM)
// //       const wav = buildWav(merged, 16000);
// //       const fd = new FormData(); fd.append("audio", wav, "rec.wav");
// //       const {data} = await axios.post(`${API}/voice_search`, fd, {timeout:30000});
// //       if (data.success && data.transcript) { setState("idle"); onResult(data.transcript); }
// //       else { setState("error"); onError?.(data.error||"No speech detected — speak clearly and try again"); setTimeout(()=>setState("idle"),4000); }
// //     } catch(err) {
// //       const detail = err?.response?.data?.detail || err.message || "";
// //       setState("error");
// //       onError?.(err?.response?.status===503 ? "Vosk model not loaded" : detail.slice(0,60) || "Voice failed");
// //       setTimeout(()=>setState("idle"),5000);
// //     }
// //   }, [state]);

// //   return { state, ok, start, stop };
// // }

// // /* ══════════════════════════════════════════════════════════════════════════ */
// // export default function App() {
// //   const [view,        setView]        = useState("timeline");
// //   const [images,      setImages]      = useState([]);
// //   const [faces,       setFaces]       = useState([]);
// //   const [albums,      setAlbums]      = useState([]);
// //   const [dupes,       setDupes]       = useState([]);
// //   const [trash,       setTrash]       = useState([]);
// //   const [stats,       setStats]       = useState(null);
// //   const [loading,     setLoading]     = useState(false);
// //   const [searchQ,     setSearchQ]     = useState("");
// //   const [searchMode,  setSearchMode]  = useState("text");
// //   const [colorPick,   setColorPick]   = useState("blue");
// //   const [hybridFile,  setHybridFile]  = useState(null);
// //   const [hybridPrev,  setHybridPrev]  = useState(null);
// //   const [voiceErr,    setVoiceErr]    = useState("");
// //   const [reindexMsg,  setReindexMsg]  = useState("");
// //   const [reindexing,  setReindexing]  = useState(false);

// //   /* New feature state */
// //   const [emoSummary,    setEmoSummary]    = useState([]);
// //   const [emoFilter,     setEmoFilter]     = useState("");        // active emotion filter
// //   const [onThisDay,     setOnThisDay]     = useState(null);
// //   const [otdExpanded,   setOtdExpanded]   = useState(false);

// //   const [batchMode,     setBatchMode]     = useState(false);
// //   const [selected,      setSelected]      = useState(new Set());
// //   const [batchTagInput, setBatchTagInput] = useState("");
// //   const [batchAlbumId,  setBatchAlbumId]  = useState("");

// //   const [allTags,       setAllTags]       = useState([]);
// //   const [activeTag,     setActiveTag]     = useState("");

// //   const [createAlbumOpen, setCreateAlbumOpen] = useState(false);
// //   const [newAlbumTitle,   setNewAlbumTitle]   = useState("");
// //   const [newAlbumDesc,    setNewAlbumDesc]    = useState("");
// //   const [pendingAlbumImages, setPendingAlbumImages] = useState([]); // snapshot of selected at open time

// //   // Named event creation
// //   const [createEventOpen, setCreateEventOpen] = useState(false);
// //   const [newEventTitle,   setNewEventTitle]   = useState("");
// //   const [newEventType,    setNewEventType]    = useState("Other");
// //   const [newEventDate,    setNewEventDate]    = useState("");
// //   const [newEventDesc,    setNewEventDesc]    = useState("");

// //   // Advanced filters
// //   const [minPeople,     setMinPeople]     = useState(0);
// //   const [coPersonIds,   setCoPersonIds]   = useState([]);
// //   const [showCoModal,   setShowCoModal]   = useState(false);

// //   // Face similarity
// //   const [faceSrcFile,   setFaceSrcFile]   = useState(null);
// //   const [faceSrcPrev,   setFaceSrcPrev]   = useState(null);
// //   const [showFaceSim,   setShowFaceSim]   = useState(false);

// //   // People frequency
// //   const [peopleFreq,    setPeopleFreq]    = useState([]);
// //   // People suggestions shown when search returns no results
// //   const [peopleSuggestions, setPeopleSuggestions] = useState([]);

// //   /* Lightbox */
// //   const [lb, setLb] = useState(null);
// //   /* Modals */
// //   const [personModal, setPersonModal] = useState(null);
// //   const [albumModal,  setAlbumModal]  = useState(null);

// //   const uploadRef = useRef(null);
// //   const voice = useVoice(
// //     t => { setSearchQ(t); setSearchMode("text"); setVoiceErr(""); runSearch("text",t); },
// //     setVoiceErr
// //   );

// //   /* keyboard */
// //   useEffect(() => {
// //     const h = e => {
// //       if (!lb) return;
// //       if (e.key==="ArrowRight") shiftLb(1);
// //       if (e.key==="ArrowLeft")  shiftLb(-1);
// //       if (e.key==="Escape")     setLb(null);
// //     };
// //     window.addEventListener("keydown",h);
// //     return ()=>window.removeEventListener("keydown",h);
// //   }, [lb]);

// //   const shiftLb = useCallback(d=>setLb(p=>p?{...p,idx:(p.idx+d+p.list.length)%p.list.length}:null),[]);
// //   const openLb  = (list,idx)=>setLb({list,idx});
// //   const closeLb = ()=>setLb(null);

// //   /* load */
// //   useEffect(()=>{ load(); },[view]);
// //   useEffect(()=>{
// //     loadEmoSummary(); loadTags(); loadTrashCount(); loadPeopleFreq();

// //   }, []);
// //   useEffect(()=>{ if(view==="timeline") loadOnThisDay(); }, [view]);

// //   const load = async () => {
// //     setLoading(true);
// //     try {
// //       if (view==="timeline")   { const r=await axios.get(`${API}/timeline`);              setImages(r.data.results||[]); }
// //       if (view==="favorites")  { const r=await axios.get(`${API}/favorites`);             setImages(r.data.results||[]); }
// //       if (view==="explore")    { const r=await axios.get(`${API}/explore/random?count=24`);setImages(r.data.results||[]); }
// //       if (view==="faces")      { const r=await axios.get(`${API}/faces`);                 setFaces(r.data.results||[]); }
// //       if (view==="albums")     { const r=await axios.get(`${API}/albums`);                setAlbums(r.data.results||[]); }
// //       if (view==="duplicates") { const r=await axios.get(`${API}/duplicates`);            setDupes(r.data.duplicate_groups||[]); }
// //       if (view==="trash")      { const r=await axios.get(`${API}/trash`);                 setTrash(r.data.results||[]); }
// //       if (view==="stats")      { const r=await axios.get(`${API}/stats`);                 setStats(r.data); }
// //     } catch {}
// //     setSelected(new Set()); setBatchMode(false);
// //     setLoading(false);
// //   };

// //   const loadEmoSummary = async () => {
// //     try { const r=await axios.get(`${API}/emotions/summary`); setEmoSummary(r.data.emotions||[]); } catch {}
// //   };
// //   const loadOnThisDay = async () => {
// //     try { const r=await axios.get(`${API}/on-this-day`); setOnThisDay(r.data); } catch {}
// //   };
// //   const loadTags = async () => {
// //     try { const r=await axios.get(`${API}/tags`); setAllTags(r.data.tags||[]); } catch {}
// //   };
// //   const loadTrashCount = async () => {
// //     try { const r=await axios.get(`${API}/trash`); setTrash(r.data.results||[]); } catch {}
// //   };

// //   const loadPeopleFreq = async () => {
// //     try { const r=await axios.get(`${API}/people/frequency`); setPeopleFreq(r.data.people||[]); } catch {}
// //   };

// //   // Group photo filter
// //   const filterGroupPhotos = async (n) => {
// //     setMinPeople(n); setLoading(true); setView("search");
// //     try { const r=await axios.get(`${API}/group-photos?min_people=${n}`); setImages(r.data.results||[]); }
// //     catch {} finally { setLoading(false); }
// //   };

// //   // Co-occurrence search
// //   const runCoOccurrence = async (ids) => {
// //     if (!ids||ids.length<2) return;
// //     setLoading(true); setView("search"); setShowCoModal(false);
// //     try { const r=await axios.get(`${API}/co-occurrence?person_ids=${ids.join(",")}`); setImages(r.data.results||[]); }
// //     catch {} finally { setLoading(false); }
// //   };

// //   // Face similarity
// //   const runFaceSimilarity = async () => {
// //     if (!faceSrcFile) return;
// //     setLoading(true); setView("search"); setShowFaceSim(false);
// //     const fd=new FormData(); fd.append("file", faceSrcFile); fd.append("top_k","30");
// //     try { const r=await axios.post(`${API}/face-similarity`, fd); setImages(r.data.results||[]); }
// //     catch(err) { alert("Face search failed: "+(err?.response?.data?.detail||err.message)); }
// //     finally { setLoading(false); setFaceSrcFile(null); setFaceSrcPrev(null); }
// //   };

// //   const runSearch = async (mode=searchMode, q=searchQ) => {
// //     setEmoFilter(""); setActiveTag(""); setLoading(true); setView("search");
// //     try {
// //       const fd=new FormData(); let res;
// //       if (mode==="text")     { fd.append("query",q);          res=await axios.post(`${API}/search`,fd); }
// //       if (mode==="describe") { fd.append("description",q);    res=await axios.post(`${API}/search/describe`,fd); }
// //       if (mode==="image")    { fd.append("file",hybridFile); fd.append("top_k",20); res=await axios.post(`${API}/search/image`,fd); }
// //       if (mode==="hybrid")   { fd.append("query",q); if(hybridFile)fd.append("file",hybridFile); fd.append("top_k",20); res=await axios.post(`${API}/search/hybrid`,fd); }
// //       if (mode==="color")    { fd.append("color",colorPick); fd.append("top_k",20); res=await axios.post(`${API}/search/color`,fd); }
// //       let results = res?.data?.results || [];
// //       // If text search returned nothing, try person name search as fallback
// //       // Only for queries that look like person names (not animals/objects/descriptive)
// //       const _NON_PERSON = new Set([
// //         "cat","dog","horse","cow","bird","fish","fox","lion","tiger","bear","wolf","pig",
// //         "rabbit","duck","frog","snake","deer","sheep","goat","chicken","otter","seal",
// //         "panda","koala","monkey","giraffe","zebra","elephant","whale","shark","eagle",
// //         "owl","penguin","parrot","bee","ant","spider","butterfly","kitten","puppy",
// //         "sunset","beach","ocean","mountain","forest","river","flower","tree","sky",
// //         "snow","rain","night","city","park","road","bridge","building","house","lake",
// //         "football","soccer","cricket","basketball","tennis","volleyball","swimming",
// //         "running","cycling","boxing","golf","hockey","baseball","rugby","sport","sports",
// //         "game","match","player","athlete","stadium","gym","yoga","dance","exercise",
// //         "car","bus","train","plane","boat","ship","bike","truck",
// //         "food","pizza","cake","coffee","burger","sushi","pasta","bread",
// //         "sunrise","rainbow","storm","cloud","wind","landscape","nature","sky",
// //       ]);
// //       const _qWords = q.trim().toLowerCase().split(/\s+/);
// //       const _isPersonQuery = _qWords.length <= 4 &&
// //         _qWords.some(w => w.length >= 3 && !_NON_PERSON.has(w)) &&
// //         !_qWords.every(w => ["a","an","the","in","on","at","of","for","with",
// //           "woman","man","girl","boy","black","white","red","blue","green","yellow",
// //           "dress","shirt","jacket","hair","wearing","standing"].includes(w));
// //       if (mode==="text" && results.length===0 && q.trim().length>=2 && _isPersonQuery) {
// //         try {
// //           const pr = await axios.get(`${API}/people/search?q=${encodeURIComponent(q.trim())}`);
// //           if (pr.data.results?.length) {
// //             results = pr.data.results.flatMap(p => p.results || []);
// //           }
// //         } catch {}
// //       }
// //       // Show people suggestions if still no results
// //       if (mode==="text" && results.length===0) {
// //         const suggestions = res?.data?.people_suggestions || [];
// //         setPeopleSuggestions(suggestions);
// //       } else {
// //         setPeopleSuggestions([]);
// //       }
// //       setImages(results);
// //     } catch { setImages([]); }
// //     setLoading(false);
// //   };

// //   const filterByEmotion = async (emo) => {
// //     if (emoFilter===emo) { setEmoFilter(""); load(); return; }
// //     setEmoFilter(emo); setView("search"); setLoading(true);
// //     try { const fd=new FormData(); fd.append("emotion",emo); const r=await axios.post(`${API}/search/emotion`,fd); setImages(r.data.results||[]); }
// //     catch { setImages([]); }
// //     setLoading(false);
// //   };

// //   const filterByTag = async (tag) => {
// //     if (activeTag===tag) { setActiveTag(""); load(); return; }
// //     setActiveTag(tag); setView("timeline"); setLoading(true);
// //     try { const r=await axios.get(`${API}/tags/${tag}/images`); setImages(r.data.results||[]); }
// //     catch { setImages([]); }
// //     setLoading(false);
// //   };

// //   const handleUpload = async (files) => {
// //     setLoading(true);
// //     for (const f of Array.from(files)) { try { const fd=new FormData(); fd.append("file",f); await axios.post(`${API}/upload`,fd); } catch {} }
// //     load();
// //   };

// //   const doReindex = async () => {
// //     setReindexing(true); setReindexMsg("Clustering…");
// //     try { const r=await axios.post(`${API}/recluster`); setReindexMsg(`✓ ${r.data.people} people · ${r.data.albums} albums`); setTimeout(load,500); }
// //     catch { setReindexMsg("Failed"); }
// //     setReindexing(false);
// //   };

// //   const reprocessEmotions = async () => {
// //     setReindexing(true); setReindexMsg("Reprocessing emotions…");
// //     try {
// //       const r = await axios.post(`${API}/reprocess-emotions`);
// //       setReindexMsg(`✓ ${r.data.message}`);
// //       setTimeout(()=>{ loadEmoSummary(); if(view==="stats") load(); }, 3000);
// //     } catch(err) {
// //       setReindexMsg("Emotion reprocess failed: " + (err?.response?.data?.detail || err.message));
// //     }
// //     setReindexing(false);
// //   };

// //   const reprocessColors = async () => {
// //     setReindexing(true); setReindexMsg("Recomputing colors…");
// //     try {
// //       const r = await axios.post(`${API}/reprocess-colors`);
// //       setReindexMsg(`✓ Colors updated for ${r.data.updated} photos`);
// //     } catch(err) {
// //       setReindexMsg("Color reprocess failed: " + (err?.response?.data?.detail || err.message));
// //     }
// //     setReindexing(false);
// //   };

// //   const reprocessNames = async () => {
// //     setReindexing(true); setReindexMsg("Auto-naming people…");
// //     try {
// //       const r = await axios.post(`${API}/reprocess-names`);
// //       setReindexMsg(`✓ ${r.data.message}`);
// //       setTimeout(()=>{ if(view==="faces") load(); }, 1000);
// //     } catch(err) {
// //       setReindexMsg("Name detection failed: " + (err?.response?.data?.detail || err.message));
// //     }
// //     setReindexing(false);
// //   };

// //   const reprocessCaptions = async (forceAll=false) => {
// //     setReindexing(true); setReindexMsg(forceAll ? "Re-captioning all photos…" : "Captioning photos missing descriptions…");
// //     try {
// //       const r = await axios.post(`${API}/recaption${forceAll ? "?force_all=true" : ""}`);
// //       if (r.data.status === "nothing_to_do") {
// //         setReindexMsg("✓ All photos already have captions. Use 'Re-caption All' to force.");
// //       } else {
// //         setReindexMsg(`✓ ${r.data.message}`);
// //       }
// //     } catch(err) {
// //       setReindexMsg("Caption failed: " + (err?.response?.data?.detail || err.message));
// //     }
// //     setReindexing(false);
// //   };

// //   const toggleFav = async (id, e) => {
// //     e?.stopPropagation();
// //     try { const fd=new FormData(); fd.append("image_id",id); await axios.post(`${API}/toggle_favorite`,fd); if(view==="favorites")load(); } catch {}
// //   };

// //   const softDelete = async (id, e) => {
// //     e?.stopPropagation();
// //     if (!window.confirm("Move to trash?")) return;
// //     try {
// //       const fd=new FormData(); fd.append("image_id",id);
// //       await axios.post(`${API}/delete_image`,fd);
// //       setImages(p=>p.filter(i=>i.id!==id));
// //       loadTrashCount();
// //     } catch(err) {
// //       alert("Could not move to trash: " + (err?.response?.data?.detail || err.message));
// //     }
// //   };

// //   const restoreImg = async id => {
// //     try {
// //       const fd=new FormData(); fd.append("image_id",id);
// //       await axios.post(`${API}/restore`,fd);
// //       setTrash(p=>p.filter(i=>i.id!==id));
// //     } catch(err) {
// //       alert("Restore failed: " + (err?.response?.data?.detail || err.message));
// //     }
// //   };
// //   const permDelete = async id => {
// //     if (!window.confirm("Delete forever? This cannot be undone.")) return;
// //     try {
// //       const fd=new FormData(); fd.append("image_id",id);
// //       await axios.post(`${API}/permanent_delete`,fd);
// //       setTrash(p=>p.filter(i=>i.id!==id));
// //     } catch(err) {
// //       alert("Delete failed: " + (err?.response?.data?.detail || err.message));
// //     }
// //   };

// //   const openPerson = async p => {
// //     setPersonModal({...p,images:null});
// //     try { const r=await axios.get(`${API}/people/${p.id}`); setPersonModal(r.data); } catch {}
// //   };
// //   const openAlbum = async a => {
// //     setAlbumModal({...a,images:null});
// //     try { const r=await axios.get(`${API}/albums/${a.id}`); setAlbumModal(r.data); } catch {}
// //   };
// //   const renamePerson = async (id,name) => {
// //     try { const fd=new FormData(); fd.append("name",name); await axios.post(`${API}/people/${id}`,fd); load(); setPersonModal(p=>p&&{...p,name}); } catch {}
// //   };

// //   /* Batch actions */
// //   const toggleSelect = useCallback((id) => {
// //     setSelected(prev => { const s=new Set(prev); s.has(id)?s.delete(id):s.add(id); return s; });
// //   }, []);
// //   const selectAll = () => setSelected(new Set(images.map(i=>i.id)));
// //   const clearSel  = () => setSelected(new Set());

// //   const batchFavorite = async (val) => {
// //     if (!selected.size) return;
// //     const fd=new FormData(); fd.append("image_ids",[...selected].join(",")); fd.append("value",val?1:0);
// //     try { await axios.post(`${API}/batch/favorite`,fd); load(); } catch {}
// //   };
// //   const batchDelete = async () => {
// //     if (!selected.size||!window.confirm(`Trash ${selected.size} photos?`)) return;
// //     const fd=new FormData(); fd.append("image_ids",[...selected].join(","));
// //     try { await axios.post(`${API}/batch/delete`,fd); load(); } catch {}
// //   };
// //   const batchTag = async () => {
// //     if (!selected.size||!batchTagInput.trim()) return;
// //     const fd=new FormData(); fd.append("image_ids",[...selected].join(",")); fd.append("tag",batchTagInput.trim());
// //     try { await axios.post(`${API}/batch/tag`,fd); setBatchTagInput(""); loadTags(); } catch {}
// //   };
// //   const batchAddAlbum = async () => {
// //     if (!selected.size||!batchAlbumId) return;
// //     const fd=new FormData(); fd.append("image_ids",[...selected].join(",")); fd.append("album_id",batchAlbumId);
// //     try { await axios.post(`${API}/batch/album`,fd); setBatchAlbumId(""); load(); } catch {}
// //   };

// //   const createAlbum = async (imageIds=[]) => {
// //     if (!newAlbumTitle.trim()) return;
// //     const fd=new FormData();
// //     fd.append("title", newAlbumTitle);
// //     fd.append("description", newAlbumDesc);
// //     // Priority: explicit imageIds arg → pendingAlbumImages snapshot → current selected
// //     const ids = imageIds.length ? imageIds
// //               : pendingAlbumImages.length ? pendingAlbumImages
// //               : [...selected];
// //     console.log("Creating album with", ids.length, "images:", ids);
// //     if (ids.length) fd.append("image_ids", ids.join(","));
// //     try {
// //       const r = await axios.post(`${API}/albums/create`, fd);
// //       setCreateAlbumOpen(false); setNewAlbumTitle(""); setNewAlbumDesc("");
// //       const addedCount = ids.length;
// //       setPendingAlbumImages([]);
// //       setSelected(new Set()); setBatchMode(false);
// //       setView("albums");
// //       await load();
// //       // Open the newly created album immediately so user can see it
// //       if (r.data?.id) {
// //         const newAlbum = {id: r.data.id, title: r.data.title, type:"manual",
// //                          count: addedCount, images: null};
// //         openAlbum(newAlbum);
// //       }
// //       return r.data;
// //     } catch(err) { alert("Failed to create album: " + (err?.response?.data?.detail || err.message)); }
// //   };
// //   const deleteAlbum = async id => {
// //     try {
// //       await axios.delete(`${API}/albums/${id}/delete`);
// //       setAlbumModal(null);
// //       load();
// //     } catch(err) { alert("Delete failed: " + (err?.response?.data?.detail || err.message)); }
// //   };

// //   const renameAlbum = async (id, title, description) => {
// //     const fd=new FormData(); fd.append("title", title);
// //     if (description) fd.append("description", description);
// //     try {
// //       await axios.post(`${API}/albums/${id}/rename`, fd);
// //       load();
// //       setAlbumModal(prev => prev && prev.id===id ? {...prev, title, description: description||prev.description} : prev);
// //     } catch(err) { alert("Rename failed: " + (err?.response?.data?.detail || err.message)); }
// //   };

// //   const EVENT_TYPES = ["Birthday","Trip","Vacation","Wedding","Anniversary","Graduation","Party","Holiday","Family","Work","Other"];

// //   const createNamedEvent = async () => {
// //     if (!newEventTitle.trim()) return;
// //     const fd=new FormData();
// //     fd.append("title", newEventTitle.trim());
// //     fd.append("event_type", newEventType);
// //     fd.append("description", newEventDesc.trim());
// //     fd.append("date_str", newEventDate);
// //     try {
// //       await axios.post(`${API}/events/create`, fd);
// //       setCreateEventOpen(false);
// //       setNewEventTitle(""); setNewEventType("Other"); setNewEventDate(""); setNewEventDesc("");
// //       load();
// //     } catch(err) { alert("Could not create event: " + (err?.response?.data?.detail || err.message)); }
// //   };

// //   const addTagToImage = async (imgId, tag) => {
// //     const fd=new FormData(); fd.append("tag",tag);
// //     try {
// //       const r = await axios.post(`${API}/photo/${imgId}/tags/add`,fd);
// //       const newTags = r.data.tags || [];
// //       // Update the images list in place so lightbox reflects latest tags
// //       setImages(prev => prev.map(i => i.id===imgId ? {...i, user_tags: newTags} : i));
// //       loadTags();
// //       return newTags;
// //     } catch { return null; }
// //   };
// //   const removeTagFromImage = async (imgId, tag) => {
// //     const fd=new FormData(); fd.append("tag",tag);
// //     try {
// //       const r = await axios.post(`${API}/photo/${imgId}/tags/remove`,fd);
// //       const newTags = r.data.tags || [];
// //       setImages(prev => prev.map(i => i.id===imgId ? {...i, user_tags: newTags} : i));
// //       loadTags();
// //       return newTags;
// //     } catch { return null; }
// //   };

// //   const isGridView = ["timeline","search","favorites","explore"].includes(view);

// //   const NAV = [
// //     { id:"timeline",   icon:<Clock size={15}/>,     label:"Timeline"   },
// //     { id:"albums",     icon:<BookOpen size={15}/>,  label:"Albums"     },
// //     { id:"faces",      icon:<Users size={15}/>,     label:"People"     },
// //     { id:"favorites",  icon:<Heart size={15}/>,     label:"Favorites"  },
// //     { id:"duplicates", icon:<Copy size={15}/>,      label:"Duplicates" },
// //     { id:"explore",    icon:<Shuffle size={15}/>,   label:"Explore"    },
// //     { id:"stats",      icon:<BarChart3 size={15}/>, label:"Statistics" },
// //   ];
// //   const MODES = [
// //     { id:"text",     Icon:Search,   label:"Search"   },
// //     { id:"describe", Icon:FileText, label:"Describe" },
// //     { id:"image",    Icon:ImgIcon,  label:"By Image" },
// //     { id:"hybrid",   Icon:GitMerge, label:"Hybrid"   },
// //     { id:"color",    Icon:Palette,  label:"By Color" },
// //   ];
// //   const COLORS = ["red","orange","yellow","green","blue","purple","pink","white","black","gray","brown"];

// //   return (
// //     <>
// //       <CSS />
// //       <div className="shell">

// //         {/* ── SIDEBAR ─────────────────────────────────────────────────── */}
// //         <nav className="sidebar">
// //           <div className="brand">
// //             <div className="brand-mark"><Aperture size={16} strokeWidth={1.5}/></div>
// //             <div>
// //               <p className="brand-name">LUMINA</p>
// //               <p className="brand-sub">AI Gallery</p>
// //             </div>
// //           </div>

// //           <div className="nav-list">
// //             {NAV.map(n=>(
// //               <button key={n.id} className={`nav-btn ${view===n.id?"nav-btn--on":""}`} onClick={()=>setView(n.id)}>
// //                 <span className="nav-ic">{n.icon}</span>
// //                 <span>{n.label}</span>
// //                 {view===n.id && <span className="nav-pip"/>}
// //               </button>
// //             ))}
// //           </div>

// //           <div className="sidebar-sep"/>

// //           {/* Advanced Filters */}
// //           <p className="sidebar-section-label"><SlidersHorizontal size={10}/> FILTERS</p>
// //           <button className="nav-btn" style={{fontSize:11}} onClick={()=>filterGroupPhotos(2)}>
// //             <span className="nav-ic"><Users size={14}/></span><span>2+ People</span>
// //           </button>
// //           <button className="nav-btn" style={{fontSize:11}} onClick={()=>filterGroupPhotos(3)}>
// //             <span className="nav-ic"><Users size={14}/></span><span>3+ People</span>
// //           </button>
// //           <button className="nav-btn" style={{fontSize:11}} onClick={()=>filterGroupPhotos(5)}>
// //             <span className="nav-ic"><Users size={14}/></span><span>5+ People</span>
// //           </button>
// //           <button className="nav-btn" style={{fontSize:11}} onClick={()=>setShowCoModal(true)}>
// //             <span className="nav-ic"><GitMerge size={14}/></span><span>Co-Appear</span>
// //           </button>
// //           <button className="nav-btn" style={{fontSize:11}} onClick={()=>setShowFaceSim(true)}>
// //             <span className="nav-ic"><UserCheck size={14}/></span><span>Face Search</span>
// //           </button>

// //           <div className="sidebar-sep"/>

// //           <button className={`nav-btn nav-btn--trash ${view==="trash"?"nav-btn--on":""}`} onClick={()=>setView("trash")}>
// //             <span className="nav-ic"><Trash size={15}/></span>
// //             <span>Trash</span>
// //             {trash.length>0 && <span className="badge">{trash.length}</span>}
// //           </button>

// //           {/* Tags sidebar section */}
// //           {allTags.length>0 && (
// //             <>
// //               <div className="sidebar-sep"/>
// //               <p className="sidebar-section-label"><Hash size={10}/> TAGS</p>
// //               <div className="tag-nav-list">
// //                 {allTags.slice(0,8).map(t=>(
// //                   <button key={t.tag} className={`tag-nav-btn ${activeTag===t.tag?"tag-nav-btn--on":""}`} onClick={()=>filterByTag(t.tag)}>
// //                     <span>#{t.tag}</span><span className="tag-nav-count">{t.count}</span>
// //                   </button>
// //                 ))}
// //               </div>
// //             </>
// //           )}

// //           <div className="sidebar-sep"/>
// //           <div className="sidebar-foot">
// //             <button className="reindex-btn reindex-btn--primary" onClick={doReindex} disabled={reindexing}>
// //               <Zap size={13} className={reindexing?"spin":""}/>{reindexing?"Indexing…":"Re-index AI"}
// //             </button>
// //             <ToolsAccordion reindexing={reindexing}
// //               onEmotions={reprocessEmotions}
// //               onColors={reprocessColors}
// //               onNames={reprocessNames}
// //               onCaptions={()=>reprocessCaptions(false)}
// //               onRecaptionAll={()=>reprocessCaptions(true)}
// //               onCleanup={async()=>{ if(!window.confirm("Delete all empty albums?")) return; const r=await axios.delete(`${API}/albums/empty/cleanup`); setReindexMsg(`✓ Deleted ${r.data.deleted} empty albums`); load(); }}
// //             />
// //             {reindexMsg && <p className="reindex-msg">{reindexMsg}</p>}
// //             <div className="offline-pill"><span className="dot-green"/><span>Fully Offline</span></div>
// //             {voiceErr && voiceErr.includes("setup") && (
// //               <div style={{background:"rgba(245,158,11,.08)",border:"1px solid rgba(245,158,11,.2)",borderRadius:7,padding:"8px 10px",fontSize:10,color:"rgba(245,158,11,.9)",lineHeight:1.5}}>
// //                 <strong>Voice Setup:</strong><br/>
// //                 1. pip install vosk<br/>
// //                 2. Download model from<br/>
// //                 <span style={{color:"#60a5fa",wordBreak:"break-all"}}>alphacephei.com/vosk/models</span><br/>
// //                 3. Extract to:<br/>
// //                 <code style={{fontSize:9,color:"#aaa"}}>models/vosk-model-small-en-us</code>
// //               </div>
// //             )}
// //           </div>
// //         </nav>

// //         {/* ── MAIN ────────────────────────────────────────────────────── */}
// //         <div className="main">

// //           {/* TOPBAR */}
// //           <header className="topbar">
// //             <div className="mode-strip">
// //               {MODES.map(({id,Icon,label})=>(
// //                 <button key={id} className={`mode-btn ${searchMode===id?"mode-btn--on":""}`} onClick={()=>setSearchMode(id)}>
// //                   <Icon size={11}/> {label}
// //                 </button>
// //               ))}
// //               {voice.state!=="idle" && (
// //                 <span className={`voice-chip voice-chip--${voice.state}`}>
// //                   {voice.state==="rec"        && <><MicOff size={10}/> Recording</>}
// //                   {voice.state==="processing" && <><Loader2 size={10} className="spin"/> Transcribing…</>}
// //                   {voice.state==="error"      && <span title={voiceErr}>{voiceErr.length>38?voiceErr.slice(0,38)+"…":voiceErr}</span>}
// //                 </span>
// //               )}
// //             </div>
// //             {/* ── Quick emoji search chips ───────────────────────────────── */}
// //             <div className="emoji-quick-bar">
// //               <span className="emoji-quick-label">Quick:</span>
// //               {[["😊","happy"],["😢","sad"],["😠","angry"],["😲","surprised"],["😨","fearful"],
// //                 ["🐶","dog"],["🏖️","beach"],["🌅","sunset"],["🎉","party"],["✈️","travel"],
// //                 ["❄️","snow"],["🌸","flowers"],["🍕","food"],["🏋️","gym"],["🌊","ocean"]
// //               ].map(([emoji,label])=>(
// //                 <button key={emoji} className="emoji-quick-btn" title={label}
// //                   onClick={()=>{ setSearchQ(emoji); setSearchMode("text"); runSearch("text", emoji); }}>
// //                   {emoji}
// //                 </button>
// //               ))}
// //             </div>
// //             <div className="search-bar">
// //               {(searchMode==="text"||searchMode==="describe"||searchMode==="hybrid") && (
// //                 <div className="search-wrap">
// //                   <Search size={14} className="search-ic"/>
// //                   <input className="search-in" value={searchQ} onChange={e=>setSearchQ(e.target.value)}
// //                     onKeyDown={e=>e.key==="Enter"&&runSearch()}
// //                     placeholder={searchMode==="describe"?"Describe what's in the photo…":searchMode==="hybrid"?"Text + optional image…":"Search by scene, object, person, text…"}
// //                   />
// //                   {voice.ok && (
// //                     <button className={`mic-btn ${voice.state==="rec"?"mic-btn--rec":voice.state==="processing"?"mic-btn--proc":""}`}
// //                       onClick={voice.state==="rec"?voice.stop:voice.start}>
// //                       {voice.state==="rec"?<MicOff size={14}/>:voice.state==="processing"?<Loader2 size={14} className="spin"/>:<Mic size={14}/>}
// //                     </button>
// //                   )}
// //                 </div>
// //               )}
// //               {(searchMode==="image"||searchMode==="hybrid") && (
// //                 <label className="img-search-label">
// //                   {hybridPrev?<img src={hybridPrev} className="img-search-thumb" alt=""/>:<ImgIcon size={13}/>}
// //                   <span>{hybridPrev?"Ready":"Upload ref"}</span>
// //                   <input type="file" accept="image/*" style={{display:"none"}} onChange={e=>{const f=e.target.files?.[0];if(f){setHybridFile(f);setHybridPrev(URL.createObjectURL(f))}}}/>
// //                 </label>
// //               )}
// //               {searchMode==="color" && (
// //                 <div className="color-strip">
// //                   {COLORS.map(c=>(
// //                     <button key={c} title={c} className={`clr-dot ${colorPick===c?"clr-dot--on":""}`}
// //                       style={{background:c==="white"?"#e5e5e5":c==="gray"?"#888":c}}
// //                       onClick={()=>setColorPick(c)}/>
// //                   ))}
// //                 </div>
// //               )}
// //               <button className="btn-go" onClick={()=>runSearch()}>Search</button>
// //               {isGridView && (
// //                 <button className={`btn-batch ${batchMode?"btn-batch--on":""}`} onClick={()=>{setBatchMode(p=>!p);clearSel();}}>
// //                   <CheckSquare size={13}/> {batchMode?"Exit Select":"Select"}
// //                 </button>
// //               )}
// // <button className="btn-upload" onClick={()=>uploadRef.current?.click()}>
// //                 <Upload size={13}/> Upload
// //               </button>
// //               <input ref={uploadRef} type="file" multiple accept="image/*" style={{display:"none"}} onChange={e=>handleUpload(e.target.files)}/>
// //             </div>

// //             {/* Emotion filter bar */}
// //             {isGridView && emoSummary.length>0 && (
// //               <div className="emo-filter-bar">
// //                 <span className="emo-filter-label">Filter:</span>
// //                 {emoSummary.filter(e=>e.emotion&&e.emotion!=="neutral").map(e=>(
// //                   <button key={e.emotion}
// //                     className={`emo-filter-btn ${emoFilter===e.emotion?"emo-filter-btn--on":""}`}
// //                     style={emoFilter===e.emotion?{background:EMO_COLORS[e.emotion]+"22",borderColor:EMO_COLORS[e.emotion]+"66",color:EMO_COLORS[e.emotion]}:{}}
// //                     onClick={()=>filterByEmotion(e.emotion)}>
// //                     {EMO[e.emotion]||"😐"} {e.emotion} <span className="emo-filter-count">{e.count}</span>
// //                   </button>
// //                 ))}
// //                 {(emoFilter||activeTag) && (
// //                   <button className="emo-filter-clear" onClick={()=>{setEmoFilter("");setActiveTag("");load();}}>
// //                     <X size={10}/> Clear
// //                   </button>
// //                 )}
// //               </div>
// //             )}
// //           </header>

// //           {/* BODY */}
// //           <div className="body">
// //             <AnimatePresence mode="wait">
// //               {loading ? (
// //                 <Fade key="load">
// //                   <div className="loader-wrap"><div className="loader-ring"/><p className="loader-text">Loading…</p></div>
// //                 </Fade>

// //               ) : isGridView ? (
// //                 <Fade key={view+emoFilter+activeTag}>
// //                   {/* On This Day banner */}
// //                   {view==="timeline" && onThisDay?.total>0 && (
// //                     <OnThisDayBanner data={onThisDay} expanded={otdExpanded} onToggle={()=>setOtdExpanded(p=>!p)} onOpen={openLb}/>
// //                   )}

// //                   <div className="page-head">
// //                     <h1 className="page-title">
// //                       {view==="search"?`Results for "${searchQ}"`:view==="favorites"?"Favorites":view==="explore"?"Explore":
// //                        activeTag?`#${activeTag}`:emoFilter?`${EMO[emoFilter]} ${emoFilter}`:"Timeline"}
// //                     </h1>
// //                     {images.length>0 && <span className="page-count">{images.length} photos</span>}
// //                     {batchMode && selected.size>0 && <span className="sel-count">{selected.size} selected</span>}
// //                     {view==="explore" && <button className="btn-sm" style={{marginLeft:"auto"}} onClick={load}><Shuffle size={11}/> Shuffle</button>}
// //                   </div>

// //                   {images.length===0
// //                     ? (view==="search" && peopleSuggestions.length>0 ? (
// //                         <div>
// //                           <Empty icon={ImageOff} msg="No photos matched" sub="But these people are in your gallery — click to view their photos"/>
// //                           <div style={{marginTop:16}}>
// //                             <p style={{fontSize:11,color:"var(--mu)",marginBottom:10,fontFamily:"var(--mono)",letterSpacing:".08em"}}>PEOPLE IN YOUR GALLERY:</p>
// //                             <div className="people-grid">
// //                               {peopleSuggestions.map(p=>(
// //                                 <motion.div key={p.id} className="person-card" whileHover={{y:-3}}
// //                                   onClick={async()=>{
// //                                     setLoading(true); setView("search"); setPeopleSuggestions([]);
// //                                     try { const r=await axios.get(`${API}/people/${p.id}`); setImages(r.data.results||r.data.images||[]); }
// //                                     catch {} finally { setLoading(false); }
// //                                   }}>
// //                                   <div className="person-avatar">
// //                                     {p.cover?<img src={imgUrl(p.cover)} alt="" onError={e=>e.target.src=BLANK}/>:<Users size={26} strokeWidth={1} color="#444"/>}
// //                                   </div>
// //                                   <p className="person-name">{p.name}</p>
// //                                   <p className="person-count" style={{color:"var(--ac2)",fontSize:9}}>click to rename</p>
// //                                 </motion.div>
// //                               ))}
// //                             </div>
// //                             <p style={{fontSize:11,color:"#555",marginTop:14,padding:"10px 14px",background:"var(--s2)",borderRadius:8,border:"1px solid var(--br)"}}>
// //                               💡 <strong>Tip:</strong> Click a person above → click ✏ Rename → type their real name → search will work instantly
// //                             </p>
// //                           </div>
// //                         </div>
// //                       ) : <Empty icon={ImageOff} msg={view==="search"?"No results":"No photos yet"} sub={view!=="search"?"Upload photos to get started":"Try different keywords"} onClick={view!=="search"?()=>uploadRef.current?.click():null}/>)
// //                     : <ImgGrid list={images} onOpen={i=>openLb(images,i)} onFav={toggleFav} onDel={softDelete} batchMode={batchMode} selected={selected} onToggle={toggleSelect}/>
// //                   }
// //                 </Fade>

// //               ) : view==="faces" ? (
// //                 <Fade key="faces">
// //                   <PageHead title="People" count={faces.length}/>
// //                   {faces.length===0
// //                     ? <Empty icon={Users} msg="No people detected" sub="Run Re-index to detect faces"/>
// //                     : <div className="people-grid">{faces.map(p=><PersonCard key={p.id} p={p} onClick={()=>openPerson(p)}/>)}</div>
// //                   }
// //                 </Fade>

// //               ) : view==="albums" ? (
// //                 <Fade key="albums">
// //                   <div className="page-head">
// //                     <h1 className="page-title">Albums</h1>
// //                     {albums.length>0 && <span className="page-count">{albums.length}</span>}
// //                     <button className="btn-sm btn-sm--accent" style={{marginLeft:"auto"}} onClick={()=>{ setPendingAlbumImages([]); setCreateAlbumOpen(true); }}>
// //                       <Plus size={12}/> New Album
// //                     </button>
// //                     <button className="btn-sm" style={{marginLeft:6,background:"var(--s3)",border:"1px solid var(--br)"}} onClick={()=>setCreateEventOpen(true)}>🎉 New Event</button>
// //                   </div>
// //                   {albums.length===0
// //                     ? <Empty icon={BookImage} msg="No albums" sub="Run Re-index to auto-generate or create manually"/>
// //                     : <div className="albums-grid">{albums.map(a=><AlbumCard key={a.id} a={a} onClick={()=>openAlbum(a)} onDelete={a.type==="manual"?()=>deleteAlbum(a.id):null}/>)}</div>
// //                   }
// //                 </Fade>

// //               ) : view==="duplicates" ? (
// //                 <Fade key="dupes">
// //                   <PageHead title="Duplicate Groups" count={dupes.length} unit="groups"/>
// //                   {dupes.length===0
// //                     ? <Empty icon={Copy} msg="No duplicates" sub="All images are unique"/>
// //                     : <div className="dupe-list">
// //                         {dupes.map((g,gi)=>(
// //                           <div key={gi} className="dupe-group">
// //                             <div className="dupe-head">
// //                               <span className="tag-sm">Group {gi+1}</span>
// //                               <span className="tag-sm tag-sm--accent">{g.count} duplicates</span>
// //                               {g.total_size&&<span className="tag-sm">{(g.total_size/1024/1024).toFixed(1)} MB</span>}
// //                             </div>
// //                             <ImgGrid list={g.images||[]} compact onOpen={i=>openLb(g.images,i)} onFav={toggleFav} onDel={softDelete}/>
// //                           </div>
// //                         ))}
// //                       </div>
// //                   }
// //                 </Fade>

// //               ) : view==="trash" ? (
// //                 <Fade key="trash">
// //                   <PageHead title="Trash" count={trash.length}/>
// //                   {trash.length===0 ? <Empty icon={Trash} msg="Trash is empty"/> : (
// //                     <div className="img-grid">
// //                       {trash.map(img=>(
// //                         <div key={img.id} className="trash-card">
// //                           <img src={imgUrl(img.filename)} alt="" className="trash-img" onError={e=>e.target.src=BLANK}/>
// //                           <div className="trash-overlay">
// //                             <button className="trash-btn trash-btn--restore" onClick={()=>restoreImg(img.id)}><RotateCcw size={11}/> Restore</button>
// //                             <button className="trash-btn trash-btn--del" onClick={()=>permDelete(img.id)}><AlertTriangle size={11}/> Delete Forever</button>
// //                           </div>
// //                           {img.trashed_at&&<span className="trash-date">{new Date(img.trashed_at).toLocaleDateString()}</span>}
// //                         </div>
// //                       ))}
// //                     </div>
// //                   )}
// //                 </Fade>

// //               ) : view==="stats" ? (
// //                 <Fade key="stats"><StatsPage stats={stats} peopleFreq={peopleFreq}/></Fade>
// //               ) : null}
// //             </AnimatePresence>
// //           </div>

// //           {/* BATCH ACTION BAR */}
// //           <AnimatePresence>
// //             {batchMode && (
// //               <motion.div className="batch-bar"
// //                 initial={{y:60,opacity:0}} animate={{y:0,opacity:1}} exit={{y:60,opacity:0}}>
// //                 <div className="batch-bar-left">
// //                   <span className="batch-count"><CheckCheck size={14}/> {selected.size} selected</span>
// //                   <button className="batch-sm" onClick={selectAll}>All</button>
// //                   <button className="batch-sm" onClick={clearSel}>None</button>
// //                 </div>
// //                 <div className="batch-bar-actions">
// //                   <button className="batch-action-btn batch-action-btn--fav" onClick={()=>batchFavorite(true)} title="Favorite all"><Heart size={13}/> Favorite</button>
// //                   <button className="batch-action-btn batch-action-btn--del" onClick={batchDelete} title="Trash all"><Trash2 size={13}/> Trash</button>
// //                   <div className="batch-input-group">
// //                     <input className="batch-input" placeholder="Add tag…" value={batchTagInput} onChange={e=>setBatchTagInput(e.target.value)} onKeyDown={e=>e.key==="Enter"&&batchTag()}/>
// //                     <button className="batch-action-btn" onClick={batchTag}><Tag size={12}/></button>
// //                   </div>
// //                   <div className="batch-input-group">
// //                     <select className="batch-select" value={batchAlbumId} onChange={e=>setBatchAlbumId(e.target.value)}>
// //                       <option value="">Add to album…</option>
// //                       {albums.map(a=><option key={a.id} value={a.id}>{a.title}</option>)}
// //                     </select>
// //                     <button className="batch-action-btn" onClick={batchAddAlbum} title="Add to existing album"><Layers size={12}/></button>
// //                   </div>
// //                   <button className="batch-action-btn batch-action-btn--album"
// //                     onClick={()=>{
// //                       setPendingAlbumImages([...selected]); // snapshot NOW before modal opens
// //                       setCreateAlbumOpen(true);
// //                     }}
// //                     title={`Create new album with ${selected.size} selected photos`}
// //                     disabled={selected.size===0}>
// //                     <FolderPlus size={12}/> New Album
// //                   </button>
// //                 </div>
// //               </motion.div>
// //             )}
// //           </AnimatePresence>
// //         </div>
// //       </div>

// //       {/* LIGHTBOX */}
// //       <AnimatePresence>
// //         {lb && <Lightbox list={lb.list} idx={lb.idx} onClose={closeLb} onShift={shiftLb} onFav={toggleFav} onDel={(id,e)=>{softDelete(id,e);closeLb();}} onAddTag={addTagToImage} onRemoveTag={removeTagFromImage} albums={albums}/>}
// //       </AnimatePresence>

// //       {/* PERSON MODAL */}
// //       <AnimatePresence>
// //         {personModal && <PersonModal data={personModal} onClose={()=>setPersonModal(null)} onRename={renamePerson} onOpen={(list,i)=>{setPersonModal(null);setTimeout(()=>openLb(list,i),80);}}/>}
// //       </AnimatePresence>

// //       {/* ALBUM MODAL */}
// //       <AnimatePresence>
// //         {albumModal && <AlbumModal data={albumModal} onClose={()=>setAlbumModal(null)} onRename={renameAlbum} onDelete={deleteAlbum} onOpen={(list,i)=>{setAlbumModal(null);setTimeout(()=>openLb(list,i),80);}}/>}
// //       </AnimatePresence>

// //       {/* CREATE ALBUM MODAL */}
// //       <AnimatePresence>
// //         {createAlbumOpen && (
// //           <ModalWrap onClose={()=>setCreateAlbumOpen(false)}>
// //             <div className="modal-head">
// //               <h2 className="modal-title">New Album</h2>
// //               <button className="lb-hbtn lb-close" onClick={()=>setCreateAlbumOpen(false)}><X size={17}/></button>
// //             </div>
// //             {pendingAlbumImages.length>0 && (
// //               <div style={{background:"rgba(96,165,250,.08)",border:"1px solid rgba(96,165,250,.2)",borderRadius:8,padding:"8px 12px",marginBottom:12,fontSize:12,color:"rgba(96,165,250,.9)"}}>
// //                 📸 <strong>{pendingAlbumImages.length} selected photos</strong> will be added to this album
// //               </div>
// //             )}
// //             <div className="create-album-form">
// //               <input className="form-input" placeholder="Album title…" value={newAlbumTitle}
// //                 onChange={e=>setNewAlbumTitle(e.target.value)}
// //                 onKeyDown={e=>e.key==="Enter"&&newAlbumTitle.trim()&&createAlbum()}
// //                 autoFocus/>
// //               <textarea className="form-textarea" placeholder="Description (optional)…"
// //                 value={newAlbumDesc} onChange={e=>setNewAlbumDesc(e.target.value)} rows={2}/>
// //               <div style={{display:"flex",gap:8}}>
// //                 <button className="btn-sm btn-sm--primary" style={{flex:1}} onClick={()=>createAlbum()} disabled={!newAlbumTitle.trim()}>
// //                   <FolderPlus size={13}/> Create Album
// //                 </button>
// //                 <button className="btn-sm" onClick={()=>setCreateAlbumOpen(false)}>Cancel</button>
// //               </div>
// //             </div>
// //           </ModalWrap>
// //         )}
// //       </AnimatePresence>

// //       {/* CREATE NAMED EVENT MODAL */}
// //       <AnimatePresence>
// //         {createEventOpen && (
// //           <ModalWrap onClose={()=>setCreateEventOpen(false)}>
// //             <div className="modal-head">
// //               <h2 className="modal-title">🎉 New Event</h2>
// //               <button className="lb-hbtn lb-close" onClick={()=>setCreateEventOpen(false)}><X size={17}/></button>
// //             </div>
// //             <div className="create-album-form">
// //               <label style={{fontSize:11,color:"var(--mu)",marginBottom:4}}>EVENT TYPE</label>
// //               <div style={{display:"flex",flexWrap:"wrap",gap:6,marginBottom:12}}>
// //                 {["Birthday","Trip","Vacation","Wedding","Anniversary","Graduation","Party","Holiday","Family","Work","Other"].map(t=>(
// //                   <button key={t}
// //                     className={"btn-sm" + (newEventType===t?" btn-sm--primary":"")}
// //                     style={{fontSize:11,padding:"4px 10px"}}
// //                     onClick={()=>setNewEventType(t)}>
// //                     {t==="Birthday"?"🎂":t==="Trip"?"✈️":t==="Vacation"?"🏖️":t==="Wedding"?"💍":t==="Anniversary"?"💕":t==="Graduation"?"🎓":t==="Party"?"🎊":t==="Holiday"?"🎄":t==="Family"?"👨‍👩‍👧":t==="Work"?"💼":"📅"} {t}
// //                   </button>
// //                 ))}
// //               </div>
// //               <input className="form-input" placeholder="Event name (e.g. Mom's 60th Birthday)…" value={newEventTitle} onChange={e=>setNewEventTitle(e.target.value)} autoFocus/>
// //               <input className="form-input" type="date" value={newEventDate} onChange={e=>setNewEventDate(e.target.value)} style={{marginTop:8}}/>
// //               <textarea className="form-textarea" placeholder="Notes (optional)…" value={newEventDesc} onChange={e=>setNewEventDesc(e.target.value)} rows={2} style={{marginTop:8}}/>
// //               <button className="btn-sm btn-sm--primary" onClick={createNamedEvent} disabled={!newEventTitle.trim()} style={{marginTop:4}}>
// //                 🎉 Create Event
// //               </button>
// //             </div>
// //           </ModalWrap>
// //         )}
// //       </AnimatePresence>

// //       {/* CO-OCCURRENCE MODAL */}
// //       <AnimatePresence>
// //         {showCoModal && (
// //           <ModalWrap onClose={()=>setShowCoModal(false)}>
// //             <div className="modal-head">
// //               <h2 className="modal-title"><GitMerge size={16} style={{marginRight:6,verticalAlign:"middle"}}/>Co-Appearance Search</h2>
// //               <button className="lb-hbtn lb-close" onClick={()=>setShowCoModal(false)}><X size={17}/></button>
// //             </div>
// //             <CoOccurrencePanel faces={faces} onSearch={runCoOccurrence}/>
// //           </ModalWrap>
// //         )}
// //       </AnimatePresence>

// //       {/* FACE SIMILARITY MODAL */}
// //       <AnimatePresence>
// //         {showFaceSim && (
// //           <ModalWrap onClose={()=>setShowFaceSim(false)}>
// //             <div className="modal-head">
// //               <h2 className="modal-title"><UserCheck size={16} style={{marginRight:6,verticalAlign:"middle"}}/>Face Similarity Search</h2>
// //               <button className="lb-hbtn lb-close" onClick={()=>setShowFaceSim(false)}><X size={17}/></button>
// //             </div>
// //             <div className="create-album-form">
// //               <p style={{fontSize:12,color:"var(--mu)",marginBottom:12}}>Upload a face crop or photo — find all photos with the same person.</p>
// //               {faceSrcPrev && <img src={faceSrcPrev} style={{width:120,height:120,objectFit:"cover",borderRadius:8,marginBottom:12,border:"2px solid var(--ac)"}} alt="face"/>}
// //               <input type="file" accept="image/*" style={{display:"none"}} id="face-sim-input"
// //                 onChange={e=>{
// //                   const f=e.target.files[0]; if(!f) return;
// //                   setFaceSrcFile(f);
// //                   const r=new FileReader(); r.onload=ev=>setFaceSrcPrev(ev.target.result); r.readAsDataURL(f);
// //                 }}/>
// //               <button className="btn-sm" onClick={()=>document.getElementById("face-sim-input").click()}>
// //                 <Camera size={12}/> {faceSrcPrev?"Change Photo":"Upload Face Photo"}
// //               </button>
// //               {faceSrcFile && (
// //                 <button className="btn-sm btn-sm--primary" style={{marginTop:10}} onClick={runFaceSimilarity}>
// //                   <UserCheck size={12}/> Find Similar Faces
// //                 </button>
// //               )}
// //             </div>
// //           </ModalWrap>
// //         )}
// //       </AnimatePresence>
// //     </>
// //   );
// // }

// // /* ══════════════════════════════════════════════════════════════════════════ */
// // /* ON THIS DAY BANNER                                                          */
// // /* ══════════════════════════════════════════════════════════════════════════ */
// // function OnThisDayBanner({ data, expanded, onToggle, onOpen }) {
// //   return (
// //     <div className="otd-banner">
// //       <div className="otd-header" onClick={onToggle}>
// //         <div className="otd-left">
// //           <Calendar size={15} color="#f59e0b"/>
// //           <div>
// //             <span className="otd-title">On This Day — {data.date}</span>
// //             <span className="otd-sub">{data.total} photo{data.total!==1?"s":""} from {data.years?.length} year{data.years?.length!==1?"s":""} ago</span>
// //           </div>
// //         </div>
// //         <button className="otd-toggle">{expanded?<ChevronUp size={14}/>:<ChevronDown size={14}/>}</button>
// //       </div>
// //       <AnimatePresence>
// //         {expanded && (
// //           <motion.div initial={{height:0,opacity:0}} animate={{height:"auto",opacity:1}} exit={{height:0,opacity:0}}>
// //             {data.years?.map(yr=>(
// //               <div key={yr.year} className="otd-year">
// //                 <p className="otd-year-label">{yr.year} — {yr.count} photo{yr.count!==1?"s":""}</p>
// //                 <div className="otd-strip">
// //                   {yr.images.map((img,i)=>(
// //                     <div key={img.id} className="otd-thumb" onClick={()=>onOpen(yr.images,i)}>
// //                       <img src={imgUrl(img.filename)} alt="" onError={e=>e.target.src=BLANK}/>
// //                       {img.caption_short && <div className="otd-thumb-caption">{img.caption_short}</div>}
// //                     </div>
// //                   ))}
// //                 </div>
// //               </div>
// //             ))}
// //           </motion.div>
// //         )}
// //       </AnimatePresence>
// //     </div>
// //   );
// // }


// // /* ══════════════════════════════════════════════════════════════════════════ */
// // /* IMAGE GRID                                                                  */
// // /* ══════════════════════════════════════════════════════════════════════════ */
// // function ImgGrid({ list, onOpen, onFav, onDel, compact, batchMode, selected, onToggle }) {
// //   return (
// //     <div className={`img-grid${compact?" img-grid--sm":""}`}>
// //       {list.map((img,i)=>(
// //         <PhotoCard key={img.id??i} img={img} onOpen={()=>onOpen(i)} onFav={onFav} onDel={onDel}
// //           batchMode={batchMode} isSelected={selected?.has(img.id)} onToggle={onToggle}/>
// //       ))}
// //     </div>
// //   );
// // }

// // function PhotoCard({ img, onOpen, onFav, onDel, batchMode, isSelected, onToggle }) {
// //   const q=qInfo(img.quality_level); const emo=EMO[img.dominant_emotion]||"";
// //   return (
// //     <motion.div
// //       className={`photo-card ${isSelected?"photo-card--sel":""}`}
// //       whileHover={{y:-4, scale:1.02}}
// //       whileTap={{scale:0.97}}
// //       transition={{type:"spring", stiffness:380, damping:28}}
// //       onClick={batchMode?()=>onToggle(img.id):onOpen}
// //     >
// //       <img src={imgUrl(img.filename)||BLANK} alt="" className="photo-img" loading="lazy"
// //         onError={e=>{e.target.onerror=null;e.target.src=BLANK;}}/>

// //       {/* Top chips */}
// //       {img.quality_level&&img.quality_level!=="Processing..."&&(
// //         <span className="chip chip--q" style={{color:q.c,background:q.bg,borderColor:q.c+"44"}}>{img.quality_level[0]}</span>
// //       )}
// //       {emo&&img.dominant_emotion!=="neutral"&&<span className="chip chip--emo">{emo}</span>}
// //       {img.is_favorite&&<span className="fav-dot">♥</span>}

// //       {batchMode ? (
// //         <motion.div className="batch-check" initial={{scale:0.5}} animate={{scale:1}}>
// //           {isSelected
// //             ? <CheckSquare size={20} color="#7c6af7" style={{filter:"drop-shadow(0 0 6px #7c6af7aa)"}}/>
// //             : <Square size={20} color="rgba(255,255,255,.4)"/>
// //           }
// //         </motion.div>
// //       ) : (
// //         <div className="photo-hover">
// //           <div className="photo-actions">
// //             <motion.button whileHover={{scale:1.15}} whileTap={{scale:.9}}
// //               className="ph-btn ph-btn--fav" onClick={e=>{e.stopPropagation();onFav(img.id,e);}}>
// //               <Heart size={12} fill={img.is_favorite?"#fff":"none"}/>
// //             </motion.button>
// //             <motion.button whileHover={{scale:1.15}} whileTap={{scale:.9}}
// //               className="ph-btn ph-btn--del" onClick={e=>{e.stopPropagation();onDel(img.id,e);}}>
// //               <Trash2 size={12}/>
// //             </motion.button>
// //           </div>
// //           <div className="photo-info">
// //             {img.caption_short&&<p className="photo-caption">{img.caption_short}</p>}
// //             <div className="photo-meta-row">
// //               {img.quality_level&&img.quality_level!=="Processing..."&&
// //                 <span className="meta-chip" style={{color:q.c}}>◆ {img.quality_level}</span>}
// //               {img.aesthetic_score>0&&
// //                 <span className="meta-chip" style={{color:"#fbbf24"}}>★ {Number(img.aesthetic_score).toFixed(1)}</span>}
// //               {img.dominant_emotion&&img.dominant_emotion!=="neutral"&&
// //                 <span className="meta-chip">{emo} {img.dominant_emotion}</span>}
// //               {img.user_tags?.length>0&&
// //                 <span className="meta-chip" style={{color:"#86efac"}}>#{img.user_tags[0]}{img.user_tags.length>1?` +${img.user_tags.length-1}`:""}</span>}
// //             </div>
// //           </div>
// //         </div>
// //       )}
// //     </motion.div>
// //   );
// // }

// // /* ══════════════════════════════════════════════════════════════════════════ */
// // /* LIGHTBOX                                                                    */
// // /* ══════════════════════════════════════════════════════════════════════════ */
// // function Lightbox({ list, idx, onClose, onShift, onFav, onDel, onAddTag, onRemoveTag, albums }) {
// //   const img=list[idx]; if(!img) return null;
// //   const q=qInfo(img.quality_level); const emo=EMO[img.dominant_emotion]||"😐";
// //   const [tagInput, setTagInput] = useState("");
// //   const [imgTags,  setImgTags]  = useState(img.user_tags||[]);
// //   const [addAlbum, setAddAlbum] = useState("");
// //   const [note,     setNote]     = useState(img.photo_note||"");
// //   const [noteEditing, setNoteEditing] = useState(false);
// //   const [noteSaving,  setNoteSaving]  = useState(false);

// //   // Always fetch latest tags + note from server when photo changes
// //   useEffect(()=>{
// //     setTagInput(""); setNoteEditing(false);
// //     setImgTags(img.user_tags||[]);
// //     setNote(img.photo_note||"");
// //     axios.get(`${API}/photo/${img.id}/tags`)
// //       .then(r => setImgTags(r.data.tags||[]))
// //       .catch(()=>{});
// //     axios.get(`${API}/photo/${img.id}/note`)
// //       .then(r => setNote(r.data.note||""))
// //       .catch(()=>{});
// //   }, [img.id]);

// //   const saveNote = async () => {
// //     setNoteSaving(true);
// //     const fd=new FormData(); fd.append("note", note);
// //     try { await axios.post(`${API}/photo/${img.id}/note`, fd); setNoteEditing(false); }
// //     catch(e) { alert("Failed to save note"); }
// //     setNoteSaving(false);
// //   };

// //   const doAddTag = async () => {
// //     if (!tagInput.trim()) return;
// //     const newTags = await onAddTag(img.id, tagInput.trim());
// //     if (newTags) setImgTags(newTags);
// //     else setImgTags(p=>[...new Set([...p, tagInput.trim().toLowerCase()])]);
// //     setTagInput("");
// //   };
// //   const doRemoveTag = async tag => {
// //     const newTags = await onRemoveTag(img.id, tag);
// //     if (newTags) setImgTags(newTags);
// //     else setImgTags(p=>p.filter(t=>t!==tag));
// //   };
// //   const doAddToAlbum = async () => {
// //     if (!addAlbum) return;
// //     const fd=new FormData(); fd.append("image_ids",img.id);
// //     try { await axios.post(`${API}/batch/album`,fd.set?fd:(()=>{fd.append("album_id",addAlbum);return fd;})()); } catch {}
// //     setAddAlbum("");
// //   };

// //   return (
// //     <motion.div className="lb-bg" initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}} onClick={onClose}>
// //       <motion.div className="lb-box" initial={{scale:.96}} animate={{scale:1}} exit={{scale:.96}} onClick={e=>e.stopPropagation()}>
// //         <div className="lb-left">
// //           {idx>0&&<button className="lb-nav lb-nav--l" onClick={()=>onShift(-1)}><ChevronLeft size={20}/></button>}
// //           <AnimatePresence mode="wait">
// //             <motion.img key={img.id??idx} src={imgUrl(img.filename)||BLANK} alt="" className="lb-img"
// //               initial={{opacity:0,x:16}} animate={{opacity:1,x:0}} exit={{opacity:0,x:-16}} transition={{duration:.18}}
// //               onError={e=>{e.target.onerror=null;e.target.src=BLANK;}}/>
// //           </AnimatePresence>
// //           {idx<list.length-1&&<button className="lb-nav lb-nav--r" onClick={()=>onShift(1)}><ChevronRight size={20}/></button>}
// //           <div className="lb-counter">{idx+1} / {list.length}</div>
// //           <div className="lb-hint">← → · ESC</div>
// //         </div>

// //         <div className="lb-right">
// //           <div className="lb-right-head">
// //             <h3 className="lb-right-title">Details</h3>
// //             <div className="lb-head-btns">
// //               <button className="lb-hbtn" onClick={()=>onFav(img.id)} style={{color:img.is_favorite?"#f43f5e":undefined}}>
// //                 <Heart size={15} fill={img.is_favorite?"#f43f5e":"none"} stroke={img.is_favorite?"#f43f5e":"currentColor"}/>
// //               </button>
// //               <button className="lb-hbtn" onClick={e=>onDel(img.id,e)}><Trash2 size={15}/></button>
// //               <button className="lb-hbtn lb-close" onClick={onClose}><X size={17}/></button>
// //             </div>
// //           </div>

// //           <div className="lb-scroll">
// //             {img.caption_short&&(
// //               <DB icon={<Camera size={12}/>} label="AI Caption" c="#818cf8">
// //                 <p className="db-caption">{img.caption_short}</p>
// //                 {img.caption_detailed&&img.caption_detailed!==img.caption_short&&<p className="db-caption-sub">{img.caption_detailed}</p>}
// //               </DB>
// //             )}
// //             {img.quality_score>0&&(
// //               <DB icon={<TrendingUp size={12}/>} label="Quality" c={q.c}>
// //                 <div className="q-wrap">
// //                   <div className="q-circle" style={{borderColor:q.c}}>
// //                     <span style={{color:q.c,fontSize:18,fontWeight:800}}>{Math.round(img.quality_score)}</span>
// //                   </div>
// //                   <div className="q-right">
// //                     <p className="q-level" style={{color:q.c}}>{img.quality_level}</p>
// //                     {img.sharpness>0&&<><QBar label="Sharpness" val={img.sharpness} c={q.c}/><QBar label="Exposure" val={img.exposure} c={q.c}/><QBar label="Contrast" val={img.contrast} c={q.c}/><QBar label="Composition" val={img.composition} c={q.c}/></>}
// //                   </div>
// //                 </div>
// //               </DB>
// //             )}
// //             {img.aesthetic_score>0&&(
// //               <DB icon={<Star size={12}/>} label="Aesthetic" c="#fbbf24">
// //                 <div className="aes-row"><span className="aes-score">★ {Number(img.aesthetic_score).toFixed(1)}</span>{img.aesthetic_rating&&<span className="aes-rating">{img.aesthetic_rating}</span>}</div>
// //               </DB>
// //             )}
// //             {img.dominant_emotion&&img.dominant_emotion!=="neutral"&&(
// //               <DB icon={<Smile size={12}/>} label="Emotion" c={EMO_COLORS[img.dominant_emotion]||"#f9a8d4"}>
// //                 <div className="emo-row">
// //                   <span style={{fontSize:30}}>{EMO[img.dominant_emotion]}</span>
// //                   <div><p className="emo-name">{img.dominant_emotion}</p>{img.face_emotion_count>0&&<p className="emo-sub">{img.face_emotion_count} face{img.face_emotion_count>1?"s":""}</p>}</div>
// //                 </div>
// //               </DB>
// //             )}
// //             {img.ocr_text_enhanced?.trim()&&(
// //               <DB icon={<Type size={12}/>} label="Text in Photo" c="#22d3ee">
// //                 <p className="ocr-text">{img.ocr_text_enhanced}</p>
// //               </DB>
// //             )}

// //             {/* Tags block */}
// //             <DB icon={<FileText size={12}/>} label="Personal Note" c="#fcd34d">
// //               {noteEditing ? (
// //                 <div>
// //                   <textarea className="form-textarea" value={note} onChange={e=>setNote(e.target.value)}
// //                     placeholder="Write a personal note about this photo…" rows={3} autoFocus
// //                     style={{marginBottom:6,fontSize:12}}/>
// //                   <div style={{display:"flex",gap:6}}>
// //                     <button className="btn-sm btn-sm--primary" onClick={saveNote} disabled={noteSaving}>
// //                       {noteSaving?"Saving…":"Save Note"}
// //                     </button>
// //                     <button className="btn-sm" onClick={()=>setNoteEditing(false)}>Cancel</button>
// //                   </div>
// //                 </div>
// //               ) : (
// //                 <div>
// //                   {note ? <p style={{fontSize:12,color:"var(--tx)",lineHeight:1.5,marginBottom:6}}>{note}</p>
// //                         : <p style={{fontSize:11,color:"var(--mu)"}}>No note yet</p>}
// //                   <button className="btn-sm" style={{fontSize:10,padding:"3px 8px"}} onClick={()=>setNoteEditing(true)}>
// //                     ✏ {note?"Edit Note":"Add Note"}
// //                   </button>
// //                 </div>
// //               )}
// //             </DB>

// //             <DB icon={<Hash size={12}/>} label="Tags" c="#86efac">
// //               <div className="tag-chips">
// //                 {imgTags.map(t=>(
// //                   <span key={t} className="tag-chip-item">
// //                     #{t}
// //                     <button className="tag-chip-x" onClick={()=>doRemoveTag(t)}><X size={8}/></button>
// //                   </span>
// //                 ))}
// //                 {imgTags.length===0&&<span style={{fontSize:10,color:"var(--mu)"}}>No tags yet</span>}
// //               </div>
// //               <div className="tag-add-row">
// //                 <input className="tag-add-input" placeholder="Add tag…" value={tagInput} onChange={e=>setTagInput(e.target.value)} onKeyDown={e=>e.key==="Enter"&&doAddTag()}/>
// //                 <button className="tag-add-btn" onClick={doAddTag}><Plus size={12}/></button>
// //               </div>
// //             </DB>

// //             {/* Add to album */}
// //             {albums?.length>0&&(
// //               <DB icon={<Layers size={12}/>} label="Add to Album" c="#a78bfa">
// //                 <div className="tag-add-row">
// //                   <select className="batch-select" style={{flex:1}} value={addAlbum} onChange={e=>setAddAlbum(e.target.value)}>
// //                     <option value="">Choose album…</option>
// //                     {albums.map(a=><option key={a.id} value={a.id}>{a.title}</option>)}
// //                   </select>
// //                   <button className="tag-add-btn" onClick={doAddToAlbum}><Plus size={12}/></button>
// //                 </div>
// //               </DB>
// //             )}

// //             {img.scene_label&&(
// //               <DB icon={<Tag size={12}/>} label="Objects" c="#86efac">
// //                 <div className="tag-wrap">{img.scene_label.split(",").map((t,i)=><span key={i} className="obj-tag">{t.trim()}</span>)}</div>
// //               </DB>
// //             )}
// //             <DB icon={<Info size={12}/>} label="Metadata" c="#6b7280">
// //               <div className="meta-stack">
// //                 {img.timestamp&&<MRow k="Captured" v={new Date(img.timestamp).toLocaleString()}/>}
// //                 {img.width&&<MRow k="Dimensions" v={`${img.width} × ${img.height}`}/>}
// //                 {img.person_count>0&&<MRow k="People" v={`${img.person_count} detected`}/>}
// //                 {img.score>0&&<MRow k="Relevance" v={`${Math.round(img.score)}%`}/>}
// //               </div>
// //             </DB>
// //           </div>
// //         </div>
// //       </motion.div>
// //     </motion.div>
// //   );
// // }

// // function QBar({label,val,c}){ return <div className="qbar-row"><span className="qbar-label">{label}</span><div className="qbar-track"><div className="qbar-fill" style={{width:`${Math.min(100,(val||0)*100)}%`,background:c}}/></div><span className="qbar-num">{Math.round((val||0)*100)}</span></div>; }
// // function MRow({k,v}){ return <div className="mrow"><span className="mrow-k">{k}</span><span className="mrow-v">{v}</span></div>; }
// // function DB({icon,label,c,children}){ return <div className="db" style={{"--dc":c}}><div className="db-head">{React.cloneElement(icon,{color:c})}<span className="db-label">{label}</span></div><div className="db-body">{children}</div></div>; }

// // /* Person / Album components */
// // function PersonCard({p,onClick}){ return <motion.div className="person-card" whileHover={{y:-3}} onClick={onClick}><div className="person-avatar">{p.cover?<img src={imgUrl(p.cover)} alt="" onError={e=>e.target.src=BLANK}/>:<Users size={26} strokeWidth={1} color="#444"/>}</div><p className="person-name">{p.name}</p><p className="person-count">{p.count||p.face_count||0} photos</p></motion.div>; }

// // function PersonModal({data,onClose,onRename,onOpen}){
// //   const [editing, setEditing] = useState(false);
// //   const [name, setName] = useState(data.name || "");
// //   const imgs = data.images || [];
// //   const isDefault = /^Person \d+$/.test(data.name || "");

// //   const doSave = () => {
// //     if (name.trim()) { onRename(data.id, name.trim()); setEditing(false); }
// //   };

// //   return (
// //     <ModalWrap onClose={onClose} wide>
// //       <div className="modal-head">
// //         <div>
// //           <h2 className="modal-title">{data.name}</h2>
// //           <p className="modal-sub">{data.face_count||data.count||0} photos</p>
// //         </div>
// //         <div style={{display:"flex",gap:6,alignItems:"center"}}>
// //           <button className="btn-sm btn-sm--primary" onClick={()=>{setEditing(true);setName(isDefault?"":data.name);}}>
// //             ✏ Rename
// //           </button>
// //           <button className="lb-hbtn lb-close" onClick={onClose}><X size={17}/></button>
// //         </div>
// //       </div>

// //       {isDefault && !editing && (
// //         <div style={{background:"rgba(250,204,21,.07)",border:"1px solid rgba(250,204,21,.2)",borderRadius:8,padding:"10px 14px",marginBottom:14}}>
// //           <p style={{fontSize:12,color:"rgba(250,204,21,.9)"}}>
// //             💡 <strong>Give this person a name</strong> so you can search for their photos. Click <strong>✏ Rename</strong> above.
// //           </p>
// //         </div>
// //       )}

// //       {editing && (
// //         <div style={{background:"var(--s2)",border:"1px solid var(--br)",borderRadius:10,padding:"14px 16px",marginBottom:14}}>
// //           <p style={{fontSize:11,color:"var(--mu)",marginBottom:8,fontFamily:"var(--mono)",letterSpacing:".07em"}}>
// //             ENTER PERSON'S NAME
// //           </p>
// //           <div style={{display:"flex",gap:8,alignItems:"center"}}>
// //             <input
// //               className="rename-in"
// //               style={{flex:1,fontSize:15,padding:"10px 14px"}}
// //               value={name}
// //               onChange={e=>setName(e.target.value)}
// //               onKeyDown={e=>{if(e.key==="Enter")doSave(); if(e.key==="Escape")setEditing(false);}}
// //               placeholder="e.g. Vijay, Shah Rukh Khan…"
// //               autoFocus
// //             />
// //             <button className="btn-sm btn-sm--primary" style={{padding:"10px 18px",fontSize:13}} onClick={doSave} disabled={!name.trim()}>
// //               Save
// //             </button>
// //             <button className="btn-sm" onClick={()=>setEditing(false)}>Cancel</button>
// //           </div>
// //           <p style={{fontSize:10,color:"#555",marginTop:8}}>
// //             After saving, you can search for this name in the search bar to find all their photos.
// //           </p>
// //         </div>
// //       )}

// //       {data.images===null
// //         ? <div className="loader-wrap" style={{minHeight:140}}><div className="loader-ring"/></div>
// //         : imgs.length>0
// //           ? <ImgGrid list={imgs} onOpen={i=>onOpen(imgs,i)} onFav={()=>{}} onDel={()=>{}}/>
// //           : <p className="empty-sub">No photos for this person</p>
// //       }
// //     </ModalWrap>
// //   );
// // }

// // function AlbumCard({a,onClick,onDelete}){
// //   return (
// //     <motion.div className="album-card" whileHover={{y:-3,scale:1.01}} onClick={onClick}>
// //       {a.cover
// //         ? <img src={imgUrl(a.cover)} alt="" className="album-cover" onError={e=>e.target.src=BLANK}/>
// //         : <div className="album-cover-ph"><FolderPlus size={32} strokeWidth={1} color="#555"/><span style={{fontSize:10,color:"#555",marginTop:4}}>Empty</span></div>
// //       }
// //       {a.thumbnails?.length>1&&<div className="album-strip">{a.thumbnails.slice(1,5).map((fn,i)=><img key={i} src={imgUrl(fn)} alt="" className="album-strip-img" onError={e=>e.target.style.display="none"}/>)}</div>}
// //       <div className="album-info">
// //         <h3 className="album-title">{a.title}</h3>
// //         <div className="album-meta">
// //           {a.type==="manual"
// //             ? <span className="tag-sm tag-sm--manual">✎ manual</span>
// //             : <span className="tag-sm tag-sm--accent">📅 event</span>
// //           }
// //           {/* Only show date badge if it adds info beyond the title (i.e. different month/year span) */}
// //           {a.type!=="manual" && a.date && !a.title.includes(a.date.split(" ")[0]) &&
// //             <span className="tag-sm">{a.date}</span>
// //           }
// //           <span className="tag-sm" style={{
// //             color: a.count===0 ? "rgba(239,68,68,.7)" : undefined,
// //             background: a.count===0 ? "rgba(239,68,68,.08)" : undefined,
// //           }}>
// //             {a.count===0 ? "⚠ empty" : `📸 ${a.count}`}
// //           </span>
// //         </div>
// //         {a.count===0 && a.type==="manual" && (
// //           <p style={{fontSize:9,color:"rgba(96,165,250,.6)",marginTop:4,fontFamily:"var(--mono)"}}>
// //             select photos → batch bar → add to album
// //           </p>
// //         )}
// //       </div>
// //       {onDelete&&<button className="album-del-btn" onClick={e=>{e.stopPropagation();onDelete();}} title="Delete album"><X size={12}/></button>}
// //     </motion.div>
// //   );
// // }

// // function AlbumModal({data,onClose,onOpen,onRename,onDelete}){
// //   const imgs=data.images||[];
// //   const [editing,setEditing]=React.useState(false);
// //   const [title,setTitle]=React.useState(data.title||"");
// //   const [desc,setDesc]=React.useState(data.description||"");
// //   const saveRename=()=>{ onRename&&onRename(data.id,title,desc); setEditing(false); };
// //   return (
// //     <ModalWrap onClose={onClose} wide>
// //       <div className="modal-head">
// //         <div>
// //           <h2 className="modal-title">{data.title}</h2>
// //           <p className="modal-sub">{data.image_count||data.count||0} photos{data.date?` · ${data.date}`:""}</p>
// //         </div>
// //         <div style={{display:"flex",gap:6,alignItems:"center"}}>
// //           <button className="btn-sm" onClick={()=>setEditing(e=>!e)}>✏ Rename</button>
// //           {data.type==="manual"&&onDelete&&(
// //             <button className="btn-sm" style={{color:"#f87171",borderColor:"rgba(248,113,113,.3)"}}
// //               onClick={()=>{ if(window.confirm(`Delete album "${data.title}"? Photos are kept.`)){onDelete(data.id);onClose();} }}>
// //               🗑 Delete
// //             </button>
// //           )}
// //           <button className="lb-hbtn lb-close" onClick={onClose}><X size={17}/></button>
// //         </div>
// //       </div>
// //       {editing&&(
// //         <div className="rename-row" style={{marginBottom:12}}>
// //           <input className="rename-in" value={title} onChange={e=>setTitle(e.target.value)}
// //             onKeyDown={e=>e.key==="Enter"&&saveRename()} placeholder="Album name…" autoFocus/>
// //           <input className="rename-in" value={desc} onChange={e=>setDesc(e.target.value)}
// //             placeholder="Description (optional)…" style={{marginLeft:6}}/>
// //           <button className="btn-sm btn-sm--primary" onClick={saveRename} style={{marginLeft:6}}>Save</button>
// //           <button className="btn-sm" onClick={()=>setEditing(false)} style={{marginLeft:4}}>Cancel</button>
// //         </div>
// //       )}
// //       {!editing&&data.description&&(
// //         <div className="album-desc"><span className="album-desc-lbl">DESCRIPTION</span><p>{data.description}</p></div>
// //       )}
// //       {data.images===null
// //         ? <div className="loader-wrap" style={{minHeight:140}}><div className="loader-ring"/></div>
// //         : imgs.length>0
// //           ? <ImgGrid list={imgs} onOpen={i=>onOpen(imgs,i)} onFav={()=>{}} onDel={()=>{}}/>
// //           : (
// //             <div style={{textAlign:"center",padding:"40px 20px",color:"var(--mu)"}}>
// //               <FolderPlus size={40} strokeWidth={1} style={{marginBottom:12,opacity:.4}}/>
// //               <p style={{fontSize:14,marginBottom:6}}>This album is empty</p>
// //               <p style={{fontSize:12,color:"#444"}}>
// //                 Go to <strong>Timeline</strong> → select photos (checkbox icon) →
// //                 choose this album in the batch bar → click the layers icon
// //               </p>
// //             </div>
// //           )
// //       }
// //     </ModalWrap>
// //   );
// // }

// // function ModalWrap({children,onClose,wide}){
// //   return <motion.div className="modal-bg" initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}} onClick={onClose}><motion.div className="modal-box" style={{maxWidth:wide?960:680}} initial={{scale:.95,y:12}} animate={{scale:1,y:0}} exit={{scale:.95}} onClick={e=>e.stopPropagation()}>{children}</motion.div></motion.div>;
// // }

// // function CoOccurrencePanel({faces, onSearch}) {
// //   const [selected, setSelected] = React.useState([]);
// //   const toggle = id => setSelected(p => p.includes(id) ? p.filter(x=>x!==id) : [...p, id]);
// //   return (
// //     <div className="create-album-form">
// //       <p style={{fontSize:12,color:"var(--mu)",marginBottom:10}}>Select 2+ people to find photos where they all appear together:</p>
// //       <div style={{display:"flex",flexWrap:"wrap",gap:8,marginBottom:14,maxHeight:240,overflowY:"auto"}}>
// //         {(faces||[]).map(p=>(
// //           <button key={p.id}
// //             className={"person-chip" + (selected.includes(p.id)?" person-chip--on":"")}
// //             onClick={()=>toggle(p.id)}>
// //             {p.cover && <img src={imgUrl(p.cover)} alt="" style={{width:24,height:24,borderRadius:"50%",objectFit:"cover",marginRight:6}}/>}
// //             {p.name||`Person ${p.id}`}
// //           </button>
// //         ))}
// //         {(!faces||faces.length===0) && <p style={{fontSize:12,color:"var(--mu)"}}>No named people yet. Go to People section first.</p>}
// //       </div>
// //       <button className="btn-sm btn-sm--primary" disabled={selected.length<2} onClick={()=>onSearch(selected)}>
// //         <GitMerge size={12}/> Find Co-appearing Photos ({selected.length} selected)
// //       </button>
// //     </div>
// //   );
// // }

// // function EmotionTimelineChart() {
// //   const [data, setData] = React.useState(null);
// //   React.useEffect(()=>{
// //     axios.get(`${API}/emotion-timeline`).then(r=>setData(r.data)).catch(()=>{});
// //   },[]);

// //   if (!data) return <p style={{fontSize:12,color:"var(--mu)",padding:8}}>Loading…</p>;
// //   if (!data.months?.length) return (
// //     <div style={{padding:"12px 0"}}>
// //       <p style={{fontSize:12,color:"var(--mu)"}}>No emotion timeline data yet.</p>
// //       <p style={{fontSize:11,color:"#555",marginTop:4}}>
// //         Emotions are detected in the background after uploading. Install <code style={{background:"var(--s3)",padding:"1px 5px",borderRadius:3}}>pip install fer</code> for best results, then click <strong>Re-index AI</strong>.
// //       </p>
// //     </div>
// //   );

// //   // Show server hint if only neutral detected
// //   const serverHint = data.hint;

// //   const EMO_COL = {happy:"#4ade80",sad:"#60a5fa",angry:"#f87171",neutral:"#a8a29e",surprised:"#fbbf24",disgusted:"#a78bfa",fearful:"#f9a8d4"};
// //   const EMO_EMOJI = {happy:"😊",sad:"😢",angry:"😠",neutral:"😐",surprised:"😲",disgusted:"🤢",fearful:"😨"};

// //   const months = data.months;
// //   const n = months.length;
// //   // Filter to emotions that actually have data (exclude neutral-only if others exist)
// //   const showEmos = data.emotions.filter(e => (data.series[e]||[]).some(v=>v>0));
// //   const hasRealData = showEmos.some(e => e !== "neutral");

// //   // Chart dimensions — extra bottom padding for legend
// //   const W=560, CHART_H=160, LEG_H=28, PAD={l:36,r:16,t:12,b:32};
// //   const chartW = W - PAD.l - PAD.r;
// //   const chartH = CHART_H - PAD.t - PAD.b;
// //   const maxVal = Math.max(1, ...showEmos.flatMap(e => data.series[e]||[]));

// //   const xScale = i => PAD.l + (n <= 1 ? chartW/2 : (i / (n-1)) * chartW);
// //   const yScale = v => PAD.t + chartH - (v / maxVal) * chartH;
// //   const makePath = series => series.map((v,i) =>
// //     `${i===0?"M":"L"}${xScale(i).toFixed(1)},${yScale(v).toFixed(1)}`
// //   ).join(" ");

// //   // X labels: show at most 8, evenly spaced
// //   const maxLabels = Math.min(8, n);
// //   const labelIndices = n <= maxLabels
// //     ? months.map((_,i)=>i)
// //     : Array.from({length:maxLabels}, (_,k) => Math.round(k*(n-1)/(maxLabels-1)));

// //   return (
// //     <div style={{overflowX:"auto"}}>
// //       {(serverHint || !hasRealData) && (
// //         <p style={{fontSize:11,color:"#f59e0b",marginBottom:8,background:"rgba(245,158,11,.08)",padding:"6px 10px",borderRadius:6,border:"1px solid rgba(245,158,11,.2)"}}>
// //           ⚠️ {serverHint || 'Only "neutral" detected — run pip install fer then Re-index AI.'}
// //         </p>
// //       )}
// //       <svg viewBox={`0 0 ${W} ${CHART_H + LEG_H}`} style={{width:"100%",maxWidth:W,display:"block"}}>
// //         {/* Y grid + labels */}
// //         {[0,0.25,0.5,0.75,1].map((f,i)=>{
// //           const y = PAD.t + chartH*(1-f);
// //           return (
// //             <g key={i}>
// //               <line x1={PAD.l} x2={W-PAD.r} y1={y} y2={y} stroke="#ffffff08" strokeWidth={1}/>
// //               <text x={PAD.l-4} y={y+3} fontSize={7} fill="#555" textAnchor="end">{Math.round(maxVal*f)}</text>
// //             </g>
// //           );
// //         })}
// //         {/* Axes */}
// //         <line x1={PAD.l} x2={PAD.l} y1={PAD.t} y2={PAD.t+chartH} stroke="#333" strokeWidth={1}/>
// //         <line x1={PAD.l} x2={W-PAD.r} y1={PAD.t+chartH} y2={PAD.t+chartH} stroke="#333" strokeWidth={1}/>
// //         {/* Emotion lines + dots */}
// //         {showEmos.map(e=>(
// //           <g key={e}>
// //             <path d={makePath(data.series[e]||[])} fill="none" stroke={EMO_COL[e]||"#888"} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round"/>
// //             {(data.series[e]||[]).map((v,i)=> v>0 && (
// //               <circle key={i} cx={xScale(i)} cy={yScale(v)} r={3} fill={EMO_COL[e]||"#888"} opacity={0.85}/>
// //             ))}
// //           </g>
// //         ))}
// //         {/* X axis labels */}
// //         {labelIndices.map(i=>(
// //           <text key={i} x={xScale(i)} y={PAD.t+chartH+14} fontSize={7} fill="#555" textAnchor="middle">
// //             {months[i]}
// //           </text>
// //         ))}
// //         {/* Legend — below chart, wrapping row */}
// //         {showEmos.map((e,i)=>{
// //           const cols = Math.min(showEmos.length, 4);
// //           const lx = PAD.l + (i % cols) * 130;
// //           const ly = CHART_H + Math.floor(i/cols)*14 + 4;
// //           return (
// //             <g key={e} transform={`translate(${lx},${ly})`}>
// //               <rect width={8} height={8} fill={EMO_COL[e]||"#888"} rx={2}/>
// //               <text x={11} y={7} fontSize={9} fill="#aaa">{EMO_EMOJI[e]} {e}</text>
// //             </g>
// //           );
// //         })}
// //       </svg>
// //     </div>
// //   );
// // }

// // function PeopleFreqChart({people}) {
// //   if(!people?.length) return <p style={{fontSize:12,color:"var(--mu)",padding:8}}>No people data yet.</p>;
// //   const top = people.slice(0,12);
// //   const maxN = Math.max(...top.map(p=>p.count),1);
// //   return (
// //     <div style={{display:"flex",flexDirection:"column",gap:6}}>
// //       {top.map((p,i)=>(
// //         <div key={p.id} style={{display:"flex",alignItems:"center",gap:8}}>
// //           <span style={{fontSize:11,color:"var(--mu)",width:20,textAlign:"right"}}>{i+1}</span>
// //           <span style={{fontSize:12,color:"var(--tx)",width:120,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{p.name||`Person ${p.id}`}</span>
// //           <div style={{flex:1,height:14,background:"var(--s3)",borderRadius:4,overflow:"hidden"}}>
// //             <div style={{width:`${(p.count/maxN)*100}%`,height:"100%",background:"linear-gradient(90deg,#818cf8,#a78bfa)",borderRadius:4,transition:"width .4s"}}/>
// //           </div>
// //           <span style={{fontSize:11,color:"#818cf8",width:30,textAlign:"right"}}>{p.count}</span>
// //         </div>
// //       ))}
// //     </div>
// //   );
// // }

// // function StatsPage({stats, peopleFreq}){
// //   if(!stats) return <Empty icon={BarChart3} msg="Loading stats…"/>;
// //   const cards=[{l:"Photos",v:stats.total_images,c:"#818cf8"},{l:"Faces",v:stats.total_faces,c:"#a78bfa"},{l:"People",v:stats.total_people,c:"#f472b6"},{l:"Albums",v:stats.total_albums,c:"#fbbf24"},{l:"Favorites",v:stats.total_favorites,c:"#4ade80"},{l:"Indexed",v:stats.indexed_vectors,c:"#22d3ee"}];
// //   return (
// //     <div className="stats-wrap">
// //       <PageHead title="Statistics"/>
// //       <div className="stats-cards">{cards.map((c,i)=><motion.div key={i} className="stat-card" whileHover={{y:-3}} style={{"--sc":c.c}}><p className="stat-lbl">{c.l}</p><p className="stat-val" style={{color:c.c}}>{c.v??0}</p></motion.div>)}</div>

// //       <div className="stats-sec">
// //         <p className="stats-sec-lbl">EMOTION TIMELINE</p>
// //         <EmotionTimelineChart/>
// //       </div>

// //       {peopleFreq?.length>0&&(
// //         <div className="stats-sec">
// //           <p className="stats-sec-lbl">MOST PHOTOGRAPHED PEOPLE</p>
// //           <PeopleFreqChart people={peopleFreq}/>
// //         </div>
// //       )}

// //       {stats.color_distribution?.length>0&&<div className="stats-sec"><p className="stats-sec-lbl">COLOR DISTRIBUTION</p><div className="clr-dist">{stats.color_distribution.map((d,i)=><div key={i} className="clr-dist-item"><div className="clr-dist-dot" style={{background:d.color==="gray"?"#888":d.color==="white"?"#ddd":d.color}}/><span>{d.color}</span><span className="clr-dist-n">×{d.count}</span></div>)}</div></div>}
// //       {stats.top_user_tags?.length>0&&<div className="stats-sec"><p className="stats-sec-lbl">YOUR TAGS</p><div className="tag-cloud">{stats.top_user_tags.map((t,i)=><span key={i} className="tag-cloud-item" style={{borderColor:"#86efac44",color:"#86efac"}}>#{t.tag}<span className="tag-n" style={{color:"#86efac99"}}> ×{t.count}</span></span>)}</div></div>}
// //       {stats.top_tags?.length>0&&<div className="stats-sec"><p className="stats-sec-lbl">AI DETECTED OBJECTS</p><div className="tag-cloud">{stats.top_tags.map((t,i)=><span key={i} className="tag-cloud-item">{t.tag}<span className="tag-n"> ×{t.count}</span></span>)}</div></div>}
// //     </div>
// //   );
// // }

// // function ToolsAccordion({reindexing, onEmotions, onColors, onNames, onCaptions, onRecaptionAll, onCleanup}) {
// //   const [open, setOpen] = React.useState(false);
// //   return (
// //     <div className="tools-accordion">
// //       <button className="tools-accordion-btn" onClick={()=>setOpen(p=>!p)}>
// //         <span style={{display:"flex",alignItems:"center",gap:6}}>
// //           <SlidersHorizontal size={11}/> AI Tools
// //         </span>
// //         <motion.span animate={{rotate:open?180:0}} transition={{duration:.2}}>
// //           <ChevronDown size={11}/>
// //         </motion.span>
// //       </button>
// //       <AnimatePresence>
// //         {open && (
// //           <motion.div initial={{height:0,opacity:0}} animate={{height:"auto",opacity:1}} exit={{height:0,opacity:0}}
// //             style={{overflow:"hidden"}}>
// //             <div style={{display:"flex",flexDirection:"column",gap:4,paddingTop:4}}>
// //               {[
// //                 {label:"Fix Captions",    color:"rgba(167,139,250,.9)", bg:"rgba(167,139,250,.07)", border:"rgba(167,139,250,.2)", fn:onCaptions,     title:"Caption images that are missing descriptions"},
// //                 {label:"Re-caption All",  color:"rgba(129,140,248,.85)",bg:"rgba(129,140,248,.07)", border:"rgba(129,140,248,.2)", fn:onRecaptionAll, title:"Re-run BLIP on every image (slow — use after model upgrade)"},
// //                 {label:"Fix Emotions",    color:"rgba(250,204,21,.8)",  bg:"rgba(250,204,21,.07)",  border:"rgba(250,204,21,.2)",  fn:onEmotions,     title:"Re-detect emotions on all photos"},
// //                 {label:"Fix Colors",      color:"rgba(52,211,153,.85)", bg:"rgba(52,211,153,.07)",  border:"rgba(52,211,153,.2)",  fn:onColors,       title:"Recompute dominant colors for color search"},
// //                 {label:"Auto-Name",       color:"rgba(96,165,250,.85)", bg:"rgba(96,165,250,.07)",  border:"rgba(96,165,250,.2)",  fn:onNames,        title:"Auto-detect names from photo captions"},
// //                 {label:"Clean Albums",    color:"rgba(248,113,113,.8)", bg:"rgba(248,113,113,.07)", border:"rgba(248,113,113,.2)", fn:onCleanup,      title:"Delete empty auto-generated albums"},
// //               ].map(({label,color,bg,border,fn,title})=>(
// //                 <button key={label} disabled={reindexing}
// //                   className="reindex-btn"
// //                   title={title}
// //                   style={{background:bg,borderColor:border,color,fontSize:10,padding:"7px"}}
// //                   onClick={fn}>
// //                   {label}
// //                 </button>
// //               ))}
// //             </div>
// //           </motion.div>
// //         )}
// //       </AnimatePresence>
// //     </div>
// //   );
// // }

// // function PageHead({title,count,unit="photos",extra}){ return <div className="page-head"><h1 className="page-title">{title}</h1>{count>0&&<span className="page-count">{count} {unit}</span>}{extra&&<div style={{marginLeft:"auto"}}>{extra}</div>}</div>; }
// // function Empty({icon:Icon,msg,sub,onClick}){ return <div className={`empty ${onClick?"empty--click":""}`} onClick={onClick}><Icon size={44} strokeWidth={1} color="#333"/><p className="empty-msg">{msg}</p>{sub&&<p className="empty-sub">{sub}</p>}{onClick&&<p className="empty-cta">CLICK TO UPLOAD</p>}</div>; }
// // function Fade({children}){ return <motion.div initial={{opacity:0,y:6}} animate={{opacity:1,y:0}} exit={{opacity:0}} transition={{duration:.18}}>{children}</motion.div>; }

// // /* ══════════════════════════════════════════════════════════════════════════ */
// // /* CSS                                                                         */
// // /* ══════════════════════════════════════════════════════════════════════════ */
// // function CSS() {
// //   return <style>{`
// // @import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300..900;1,14..32,300..900&family=JetBrains+Mono:wght@400;500&display=swap');

// // *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

// // :root{
// //   /* Background layers */
// //   --bg0:#000000;
// //   --bg1:#0a0a0a;
// //   --bg2:#111111;
// //   --bg3:#161616;
// //   --bg4:#1c1c1c;

// //   /* Borders */
// //   --b1:rgba(255,255,255,.06);
// //   --b2:rgba(255,255,255,.1);
// //   --b3:rgba(255,255,255,.15);

// //   /* Accent */
// //   --ac:#6366f1;
// //   --ac-l:#818cf8;
// //   --ac-d:#4f46e5;
// //   --ac-glow:rgba(99,102,241,.25);
// //   --ac-bg:rgba(99,102,241,.08);

// //   /* Text */
// //   --t1:#ffffff;
// //   --t2:#a1a1aa;
// //   --t3:#52525b;

// //   /* Status */
// //   --green:#22c55e;
// //   --red:#ef4444;
// //   --gold:#f59e0b;
// //   --sky:#38bdf8;
// //   --rose:#f43f5e;

// //   --font:'Inter',system-ui,sans-serif;
// //   --mono:'JetBrains Mono',monospace;
// //   --r:8px;
// //   --r2:12px;
// //   --r3:16px;
// // }

// // html,body,#root{height:100%;background:var(--bg0);color:var(--t1);font-family:var(--font);overflow:hidden;-webkit-font-smoothing:antialiased}

// // ::-webkit-scrollbar{width:3px;height:3px}
// // ::-webkit-scrollbar-track{background:transparent}
// // ::-webkit-scrollbar-thumb{background:var(--b2);border-radius:3px}

// // /* ═══ LAYOUT ══════════════════════════════════════════════════════════════ */
// // .shell{display:flex;height:100vh;overflow:hidden;background:var(--bg0)}

// // /* ═══ SIDEBAR ═════════════════════════════════════════════════════════════ */
// // .sidebar{
// //   width:220px;flex-shrink:0;
// //   background:var(--bg1);
// //   border-right:1px solid var(--b1);
// //   display:flex;flex-direction:column;
// //   padding:0;overflow-y:auto;overflow-x:hidden;
// // }

// // /* Brand */
// // .brand{
// //   display:flex;align-items:center;gap:10px;
// //   padding:20px 16px 16px;
// //   border-bottom:1px solid var(--b1);
// //   flex-shrink:0;
// // }
// // .brand-mark{
// //   width:32px;height:32px;border-radius:8px;
// //   background:var(--ac);
// //   display:flex;align-items:center;justify-content:center;
// //   flex-shrink:0;color:#fff;
// //   box-shadow:0 0 0 1px rgba(99,102,241,.4),0 4px 12px rgba(99,102,241,.3);
// // }
// // .brand-name{font-size:14px;font-weight:700;letter-spacing:-.01em;color:var(--t1)}
// // .brand-sub{font-family:var(--mono);font-size:9px;color:var(--t3);letter-spacing:.06em;margin-top:1px}

// // /* Nav */
// // .nav-list{display:flex;flex-direction:column;padding:8px;gap:1px;flex-shrink:0}
// // .nav-btn{
// //   display:flex;align-items:center;gap:9px;
// //   width:100%;padding:8px 10px;
// //   border-radius:var(--r);background:transparent;border:none;
// //   font-family:var(--font);font-size:13px;font-weight:500;
// //   color:var(--t3);cursor:pointer;text-align:left;
// //   transition:color .12s,background .12s;position:relative;
// // }
// // .nav-btn:hover{background:var(--bg3);color:var(--t2)}
// // .nav-btn--on{background:var(--bg3);color:var(--t1);font-weight:600}
// // .nav-btn--on .nav-ic{color:var(--ac-l)}
// // .nav-btn--trash:hover{color:var(--red);background:rgba(239,68,68,.06)}
// // .nav-ic{display:flex;flex-shrink:0;color:inherit}
// // .nav-pip{
// //   position:absolute;right:8px;
// //   width:4px;height:4px;border-radius:50%;background:var(--ac);
// //   box-shadow:0 0 6px var(--ac);
// // }
// // .badge{
// //   margin-left:auto;background:var(--red);color:#fff;
// //   font-size:10px;font-family:var(--mono);
// //   padding:1px 5px;border-radius:20px;line-height:1.5;
// // }

// // .sidebar-sep{height:1px;background:var(--b1);margin:6px 16px;flex-shrink:0}
// // .sidebar-section-label{
// //   display:flex;align-items:center;gap:5px;
// //   font-size:10px;font-weight:600;color:var(--t3);
// //   letter-spacing:.06em;text-transform:uppercase;
// //   padding:8px 16px 4px;flex-shrink:0;
// // }

// // /* Tag nav */
// // .tag-nav-list{display:flex;flex-direction:column;padding:0 8px;gap:1px}
// // .tag-nav-btn{
// //   display:flex;align-items:center;justify-content:space-between;
// //   width:100%;padding:6px 10px;border-radius:var(--r);
// //   background:transparent;border:none;
// //   font-family:var(--mono);font-size:11px;color:var(--t3);
// //   cursor:pointer;transition:all .12s;
// // }
// // .tag-nav-btn:hover{background:var(--bg3);color:var(--green)}
// // .tag-nav-btn--on{background:rgba(34,197,94,.07);color:var(--green)}
// // .tag-nav-count{font-size:9px;color:var(--t3)}

// // /* Sidebar footer */
// // .sidebar-foot{
// //   display:flex;flex-direction:column;gap:4px;
// //   margin-top:auto;padding:12px 8px;border-top:1px solid var(--b1);
// //   flex-shrink:0;
// // }
// // .reindex-btn{
// //   display:flex;align-items:center;justify-content:center;gap:6px;
// //   background:transparent;border:1px solid var(--b1);
// //   border-radius:var(--r);padding:8px;
// //   font-family:var(--font);font-size:12px;font-weight:500;
// //   color:var(--t3);cursor:pointer;transition:all .12s;
// // }
// // .reindex-btn:hover{border-color:var(--b2);color:var(--t2);background:var(--bg3)}
// // .reindex-btn:disabled{opacity:.35;cursor:default}
// // .reindex-btn--primary{
// //   background:var(--ac-bg);border-color:rgba(99,102,241,.3);
// //   color:var(--ac-l);font-weight:600;
// // }
// // .reindex-btn--primary:hover{background:rgba(99,102,241,.15);border-color:var(--ac)}

// // .reindex-msg{font-family:var(--mono);font-size:9px;color:var(--t3);text-align:center;padding:2px 4px;line-height:1.5}

// // .offline-pill{
// //   display:flex;align-items:center;gap:6px;
// //   background:rgba(34,197,94,.05);border:1px solid rgba(34,197,94,.12);
// //   border-radius:var(--r);padding:7px 10px;
// //   font-size:11px;font-weight:500;color:rgba(34,197,94,.7);
// // }
// // .dot-green{width:6px;height:6px;border-radius:50%;background:var(--green);box-shadow:0 0 6px var(--green);flex-shrink:0}

// // /* Tools accordion */
// // .tools-accordion{border:1px solid var(--b1);border-radius:var(--r);overflow:hidden;background:var(--bg2)}
// // .tools-accordion-btn{
// //   display:flex;align-items:center;justify-content:space-between;
// //   width:100%;padding:8px 10px;background:transparent;border:none;
// //   font-family:var(--font);font-size:11px;font-weight:500;color:var(--t3);cursor:pointer;
// //   transition:color .12s;
// // }
// // .tools-accordion-btn:hover{color:var(--t2)}

// // /* ═══ MAIN ════════════════════════════════════════════════════════════════ */
// // .main{flex:1;display:flex;flex-direction:column;min-width:0;overflow:hidden}

// // /* ═══ TOPBAR ══════════════════════════════════════════════════════════════ */
// // .topbar{
// //   background:var(--bg1);
// //   border-bottom:1px solid var(--b1);
// //   padding:10px 20px;
// //   display:flex;flex-direction:column;gap:8px;flex-shrink:0;
// // }

// // /* Mode pills */
// // .mode-strip{display:flex;gap:2px;align-items:center;flex-wrap:wrap}
// // .mode-btn{
// //   display:flex;align-items:center;gap:5px;
// //   padding:5px 10px;border-radius:6px;border:none;cursor:pointer;
// //   font-family:var(--font);font-size:11px;font-weight:500;
// //   background:transparent;color:var(--t3);transition:all .12s;
// // }
// // .mode-btn:hover{background:var(--bg3);color:var(--t2)}
// // .mode-btn--on{background:var(--ac);color:#fff;font-weight:600}
// // .voice-chip{
// //   margin-left:6px;display:flex;align-items:center;gap:4px;
// //   font-family:var(--mono);font-size:9px;letter-spacing:.05em;
// //   padding:4px 10px;border-radius:6px;
// // }
// // .voice-chip--rec{background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.2);color:var(--red)}
// // .voice-chip--processing{background:var(--ac-bg);border:1px solid rgba(99,102,241,.2);color:var(--ac-l)}
// // .voice-chip--error{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.15);color:#f87171}

// // /* Search bar */
// // .search-bar{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
// // .search-wrap{
// //   flex:1;min-width:200px;
// //   display:flex;align-items:center;gap:8px;
// //   background:var(--bg2);border:1px solid var(--b1);
// //   border-radius:var(--r2);padding:0 12px;
// //   transition:border-color .15s;
// // }
// // .search-wrap:focus-within{border-color:rgba(99,102,241,.4)}
// // .search-ic{color:var(--t3);flex-shrink:0}
// // .search-in{
// //   flex:1;background:transparent;border:none;
// //   padding:9px 0;font-family:var(--font);font-size:13px;
// //   color:var(--t1);outline:none;
// // }
// // .search-in::placeholder{color:var(--t3)}
// // .mic-btn{background:transparent;border:none;cursor:pointer;color:var(--t3);padding:3px;display:flex;border-radius:5px;transition:color .12s}
// // .mic-btn:hover{color:var(--t2)}
// // .mic-btn--rec{color:var(--red)}
// // .mic-btn--proc{color:var(--ac-l)}

// // .img-search-label{
// //   display:flex;align-items:center;gap:7px;cursor:pointer;
// //   background:var(--bg2);border:1px solid var(--b1);
// //   border-radius:var(--r2);padding:8px 12px;
// //   font-size:12px;color:var(--t3);flex-shrink:0;transition:all .12s;
// // }
// // .img-search-label:hover{border-color:var(--b2);color:var(--t2)}
// // .img-search-thumb{width:20px;height:20px;border-radius:4px;object-fit:cover}

// // .color-strip{display:flex;gap:5px;align-items:center;flex-wrap:wrap}
// // .clr-dot{
// //   width:22px;height:22px;border-radius:50%;border:none;cursor:pointer;
// //   outline:2px solid transparent;outline-offset:2px;transition:all .12s;flex-shrink:0;
// // }
// // .clr-dot:hover{transform:scale(1.15)}
// // .clr-dot--on{outline-color:rgba(255,255,255,.5);transform:scale(1.2)}

// // .btn-go{
// //   background:var(--ac);color:#fff;border:none;border-radius:var(--r2);
// //   padding:8px 16px;font-family:var(--font);font-size:12px;font-weight:600;
// //   cursor:pointer;transition:background .12s;flex-shrink:0;
// // }
// // .btn-go:hover{background:var(--ac-d)}

// // .btn-batch{
// //   display:flex;align-items:center;gap:5px;
// //   background:var(--bg2);border:1px solid var(--b1);
// //   color:var(--t3);border-radius:var(--r2);
// //   padding:8px 12px;font-family:var(--font);font-size:12px;font-weight:500;
// //   cursor:pointer;transition:all .12s;flex-shrink:0;
// // }
// // .btn-batch:hover{border-color:var(--b2);color:var(--t2)}
// // .btn-batch--on{background:var(--ac-bg);border-color:rgba(99,102,241,.3);color:var(--ac-l)}

// // .btn-upload{
// //   display:flex;align-items:center;gap:5px;
// //   background:var(--bg2);border:1px solid var(--b2);
// //   color:var(--t2);border-radius:var(--r2);
// //   padding:8px 14px;font-family:var(--font);font-size:12px;font-weight:600;
// //   cursor:pointer;transition:all .12s;flex-shrink:0;
// // }
// // .btn-upload:hover{border-color:var(--b3);color:var(--t1);background:var(--bg3)}

// // /* Emotion filter bar */
// // .emo-filter-bar{display:flex;align-items:center;gap:5px;flex-wrap:wrap}
// // .emo-filter-label{font-family:var(--mono);font-size:9px;color:var(--t3);letter-spacing:.06em;flex-shrink:0}
// // .emo-filter-btn{
// //   display:flex;align-items:center;gap:4px;
// //   padding:4px 10px;border-radius:20px;
// //   border:1px solid var(--b1);background:transparent;
// //   color:var(--t3);font-family:var(--font);font-size:11px;font-weight:500;
// //   cursor:pointer;transition:all .12s;flex-shrink:0;
// // }
// // .emo-filter-btn:hover{border-color:var(--b2);color:var(--t2)}
// // .emo-filter-count{font-family:var(--mono);font-size:8px;opacity:.5;margin-left:2px}
// // .emo-filter-clear{
// //   display:flex;align-items:center;gap:4px;padding:4px 9px;
// //   border-radius:20px;border:1px solid rgba(239,68,68,.2);
// //   background:rgba(239,68,68,.06);color:var(--red);
// //   font-family:var(--font);font-size:11px;font-weight:500;cursor:pointer;transition:all .12s;
// // }

// // /* Emoji bar */
// // .emoji-quick-bar{display:flex;align-items:center;gap:3px;flex-wrap:wrap}
// // .emoji-quick-label{font-family:var(--mono);font-size:9px;color:var(--t3);letter-spacing:.06em;flex-shrink:0;margin-right:2px}
// // .emoji-quick-btn{
// //   background:var(--bg3);border:1px solid var(--b1);
// //   border-radius:6px;padding:3px 6px;font-size:14px;
// //   cursor:pointer;transition:all .12s;line-height:1;
// // }
// // .emoji-quick-btn:hover{background:var(--ac-bg);border-color:rgba(99,102,241,.3);transform:scale(1.12)}

// // /* ═══ BODY ════════════════════════════════════════════════════════════════ */
// // .body{flex:1;overflow-y:auto;padding:20px 20px 80px}
// // .page-head{display:flex;align-items:baseline;gap:10px;margin-bottom:18px;flex-wrap:wrap}
// // .page-title{font-size:18px;font-weight:700;letter-spacing:-.025em;color:var(--t1)}
// // .page-count{font-family:var(--mono);font-size:9px;color:var(--t3);letter-spacing:.05em}
// // .sel-count{font-family:var(--mono);font-size:9px;color:var(--ac-l);background:var(--ac-bg);padding:2px 8px;border-radius:20px;border:1px solid rgba(99,102,241,.2)}

// // /* On This Day */
// // .otd-banner{
// //   background:rgba(245,158,11,.04);
// //   border:1px solid rgba(245,158,11,.12);
// //   border-radius:var(--r3);margin-bottom:18px;overflow:hidden;
// // }
// // .otd-header{display:flex;align-items:center;justify-content:space-between;padding:12px 16px;cursor:pointer}
// // .otd-left{display:flex;align-items:center;gap:10px}
// // .otd-title{font-size:13px;font-weight:600;display:block}
// // .otd-sub{font-family:var(--mono);font-size:9px;color:rgba(245,158,11,.5);display:block;margin-top:1px}
// // .otd-toggle{background:none;border:none;cursor:pointer;color:var(--t3);display:flex;padding:3px;border-radius:5px}
// // .otd-year{padding:0 16px 14px}
// // .otd-year-label{font-family:var(--mono);font-size:9px;color:rgba(245,158,11,.5);letter-spacing:.06em;margin-bottom:8px;display:block}
// // .otd-strip{display:flex;gap:8px;overflow-x:auto;padding-bottom:4px}
// // .otd-thumb{flex-shrink:0;width:96px;cursor:pointer;border-radius:var(--r);overflow:hidden;border:1px solid var(--b1);transition:border-color .15s}
// // .otd-thumb:hover{border-color:rgba(245,158,11,.3)}
// // .otd-thumb img{width:100%;height:64px;object-fit:cover;display:block}
// // .otd-thumb-caption{font-size:9px;color:var(--t3);padding:4px 6px;line-height:1.4;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}

// // /* ═══ IMAGE GRID ══════════════════════════════════════════════════════════ */
// // .img-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(176px,1fr));gap:8px}
// // .img-grid--sm{grid-template-columns:repeat(auto-fill,minmax(124px,1fr));gap:6px}

// // /* Photo card */
// // .photo-card{
// //   position:relative;aspect-ratio:1;
// //   border-radius:var(--r2);overflow:hidden;cursor:pointer;
// //   background:var(--bg2);border:1px solid var(--b1);
// //   transition:border-color .15s;
// // }
// // .photo-card:hover{border-color:var(--b2)}
// // .photo-card--sel{border-color:var(--ac)!important;box-shadow:0 0 0 2px rgba(99,102,241,.3)!important}
// // .photo-img{width:100%;height:100%;object-fit:cover;display:block;transition:transform .25s}
// // .photo-card:hover .photo-img{transform:scale(1.04)}
// // .chip{position:absolute;font-family:var(--mono);font-size:8px;border-radius:4px;border:1px solid;padding:1px 5px;pointer-events:none;backdrop-filter:blur(4px)}
// // .chip--q{top:7px;left:7px}
// // .chip--emo{top:7px;right:7px;font-size:13px;border:none;background:none;padding:0}
// // .fav-dot{position:absolute;bottom:7px;right:7px;color:var(--rose);font-size:13px;pointer-events:none}
// // .batch-check{position:absolute;top:7px;left:7px}
// // .photo-hover{
// //   position:absolute;inset:0;opacity:0;transition:opacity .18s;
// //   background:linear-gradient(to top,rgba(0,0,0,.9) 0%,transparent 55%);
// //   display:flex;flex-direction:column;justify-content:flex-end;
// // }
// // .photo-card:hover .photo-hover{opacity:1}
// // .photo-actions{position:absolute;top:7px;right:7px;display:flex;gap:4px}
// // .ph-btn{
// //   width:28px;height:28px;border-radius:6px;border:none;cursor:pointer;
// //   display:flex;align-items:center;justify-content:center;transition:all .12s;
// //   backdrop-filter:blur(6px);
// // }
// // .ph-btn--fav{background:rgba(244,63,94,.7);color:#fff}
// // .ph-btn--del{background:rgba(0,0,0,.6);border:1px solid rgba(255,255,255,.1);color:#888}
// // .ph-btn--del:hover{background:rgba(239,68,68,.8);color:#fff;border-color:transparent}
// // .photo-info{padding:8px 8px 6px}
// // .photo-caption{font-size:10px;line-height:1.45;color:rgba(255,255,255,.9);margin-bottom:4px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
// // .photo-meta-row{display:flex;flex-wrap:wrap;gap:3px}
// // .meta-chip{font-family:var(--mono);font-size:7.5px;color:rgba(255,255,255,.45);background:rgba(0,0,0,.4);padding:1px 5px;border-radius:3px;text-transform:capitalize}

// // /* ═══ BATCH BAR ═══════════════════════════════════════════════════════════ */
// // .batch-bar{
// //   position:absolute;bottom:0;left:0;right:0;
// //   background:rgba(10,10,10,.97);backdrop-filter:blur(12px);
// //   border-top:1px solid var(--b1);
// //   padding:10px 20px;
// //   display:flex;align-items:center;gap:10px;flex-wrap:wrap;z-index:50;
// // }
// // .batch-bar-left{display:flex;align-items:center;gap:7px}
// // .batch-count{display:flex;align-items:center;gap:5px;font-size:12px;font-weight:600;color:var(--ac-l)}
// // .batch-sm{background:var(--bg3);border:1px solid var(--b1);color:var(--t3);border-radius:6px;padding:4px 10px;font-family:var(--font);font-size:11px;font-weight:500;cursor:pointer;transition:all .12s}
// // .batch-sm:hover{border-color:var(--b2);color:var(--t2)}
// // .batch-bar-actions{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-left:auto}
// // .batch-action-btn{
// //   display:flex;align-items:center;gap:5px;padding:6px 12px;
// //   border-radius:var(--r);border:1px solid var(--b1);
// //   background:var(--bg2);color:var(--t2);
// //   font-family:var(--font);font-size:11px;font-weight:500;cursor:pointer;transition:all .12s;
// // }
// // .batch-action-btn:hover{border-color:var(--b2);color:var(--t1);background:var(--bg3)}
// // .batch-action-btn--fav:hover{border-color:rgba(244,63,94,.3);color:var(--rose)}
// // .batch-action-btn--del:hover{border-color:rgba(239,68,68,.3);color:var(--red)}
// // .batch-action-btn--album{border-color:rgba(56,189,248,.2);color:var(--sky)}
// // .batch-action-btn--album:hover{background:rgba(56,189,248,.08);border-color:rgba(56,189,248,.3)}
// // .batch-input-group{display:flex;gap:3px;align-items:center}
// // .batch-input{background:var(--bg2);border:1px solid var(--b1);border-radius:var(--r);padding:6px 10px;font-family:var(--font);font-size:11px;color:var(--t1);outline:none;width:108px;transition:border-color .12s}
// // .batch-input:focus{border-color:rgba(99,102,241,.4)}
// // .batch-select{background:var(--bg2);border:1px solid var(--b1);border-radius:var(--r);padding:6px 10px;font-family:var(--font);font-size:11px;color:var(--t2);outline:none;cursor:pointer}
// // .batch-select:focus{border-color:rgba(99,102,241,.4)}

// // /* ═══ LIGHTBOX ════════════════════════════════════════════════════════════ */
// // .lb-bg{
// //   position:fixed;inset:0;z-index:300;
// //   background:rgba(0,0,0,.9);backdrop-filter:blur(16px);
// //   display:flex;align-items:center;justify-content:center;padding:14px;
// // }
// // .lb-box{
// //   width:100%;max-width:1100px;max-height:92vh;
// //   background:var(--bg1);border:1px solid var(--b1);
// //   border-radius:16px;display:grid;grid-template-columns:1fr 300px;
// //   overflow:hidden;box-shadow:0 24px 80px rgba(0,0,0,.8);
// // }
// // .lb-left{position:relative;background:var(--bg0);display:flex;align-items:center;justify-content:center;min-height:380px;overflow:hidden}
// // .lb-img{max-width:100%;max-height:90vh;object-fit:contain;display:block}
// // .lb-nav{
// //   position:absolute;top:50%;transform:translateY(-50%);
// //   width:38px;height:38px;border-radius:50%;border:1px solid var(--b2);
// //   cursor:pointer;background:rgba(0,0,0,.5);backdrop-filter:blur(6px);
// //   color:#fff;display:flex;align-items:center;justify-content:center;
// //   transition:all .15s;z-index:2;
// // }
// // .lb-nav:hover{background:var(--ac);border-color:transparent}
// // .lb-nav--l{left:12px}
// // .lb-nav--r{right:12px}
// // .lb-counter{position:absolute;bottom:12px;left:50%;transform:translateX(-50%);font-family:var(--mono);font-size:9px;color:rgba(255,255,255,.35);background:rgba(0,0,0,.5);padding:2px 10px;border-radius:20px}
// // .lb-hint{position:absolute;bottom:12px;right:14px;font-family:var(--mono);font-size:7.5px;color:rgba(255,255,255,.15);letter-spacing:.05em}
// // .lb-right{border-left:1px solid var(--b1);background:var(--bg1);display:flex;flex-direction:column;min-height:0}
// // .lb-right-head{display:flex;align-items:center;justify-content:space-between;padding:14px 16px;border-bottom:1px solid var(--b1);flex-shrink:0}
// // .lb-right-title{font-size:13px;font-weight:600;letter-spacing:-.01em}
// // .lb-head-btns{display:flex;align-items:center;gap:3px}
// // .lb-hbtn{background:none;border:none;cursor:pointer;color:var(--t3);padding:6px;border-radius:6px;display:flex;align-items:center;transition:all .12s}
// // .lb-hbtn:hover{background:var(--bg3);color:var(--t2)}
// // .lb-close:hover{color:var(--red)}
// // .lb-scroll{flex:1;overflow-y:auto;padding:12px 14px;display:flex;flex-direction:column;gap:7px}

// // /* Detail blocks */
// // .db{background:var(--bg2);border:1px solid var(--b1);border-radius:var(--r2);overflow:hidden;border-left:2px solid var(--dc,var(--ac))}
// // .db-head{display:flex;align-items:center;gap:6px;padding:7px 11px;border-bottom:1px solid var(--b1)}
// // .db-label{font-family:var(--mono);font-size:8px;color:var(--dc,var(--ac-l));letter-spacing:.08em;font-weight:500;text-transform:uppercase}
// // .db-body{padding:9px 11px}
// // .db-caption{font-size:12px;line-height:1.55;color:var(--t1)}
// // .db-caption-sub{font-size:11px;line-height:1.55;color:var(--t2);margin-top:4px}
// // .q-wrap{display:flex;align-items:flex-start;gap:10px}
// // .q-circle{width:42px;height:42px;border-radius:50%;border:2px solid;display:flex;align-items:center;justify-content:center;flex-shrink:0}
// // .q-right{flex:1}
// // .q-level{font-size:12px;font-weight:600;margin-bottom:6px}
// // .qbar-row{display:flex;align-items:center;gap:7px;margin-bottom:4px}
// // .qbar-label{font-family:var(--mono);font-size:7.5px;color:var(--t3);width:65px;flex-shrink:0}
// // .qbar-track{flex:1;height:2px;background:rgba(255,255,255,.07);border-radius:2px;overflow:hidden}
// // .qbar-fill{height:100%;border-radius:2px}
// // .qbar-num{font-family:var(--mono);font-size:7.5px;color:var(--t3);width:22px;text-align:right}
// // .aes-row{display:flex;align-items:center;gap:9px}
// // .aes-score{font-size:24px;font-weight:800;color:var(--gold)}
// // .aes-rating{font-size:11px;color:var(--t2)}
// // .emo-row{display:flex;align-items:center;gap:10px}
// // .emo-name{font-size:12px;font-weight:600;text-transform:capitalize}
// // .emo-sub{font-family:var(--mono);font-size:8px;color:var(--t3);margin-top:2px}
// // .ocr-text{font-family:var(--mono);font-size:9px;color:var(--t2);line-height:1.6;white-space:pre-wrap;max-height:84px;overflow-y:auto}
// // .tag-wrap{display:flex;flex-wrap:wrap;gap:4px}
// // .obj-tag{font-family:var(--mono);font-size:8.5px;background:var(--bg3);color:var(--t3);padding:2px 7px;border-radius:4px;border:1px solid var(--b1)}
// // .meta-stack{display:flex;flex-direction:column;gap:5px}
// // .mrow{display:flex;justify-content:space-between;align-items:baseline;gap:12px}
// // .mrow-k{font-family:var(--mono);font-size:8px;color:var(--t3);flex-shrink:0}
// // .mrow-v{font-size:11px;color:var(--t1);text-align:right}

// // /* Tags */
// // .tag-chips{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px}
// // .tag-chip-item{display:flex;align-items:center;gap:3px;background:rgba(34,197,94,.07);border:1px solid rgba(34,197,94,.15);color:var(--green);font-family:var(--mono);font-size:8.5px;padding:2px 7px;border-radius:4px}
// // .tag-chip-x{background:none;border:none;cursor:pointer;color:rgba(34,197,94,.4);display:flex;padding:0;margin-left:2px;transition:color .12s}
// // .tag-chip-x:hover{color:var(--green)}
// // .tag-add-row{display:flex;gap:4px;align-items:center}
// // .tag-add-input{flex:1;background:var(--bg3);border:1px solid var(--b1);border-radius:6px;padding:5px 9px;font-family:var(--mono);font-size:10px;color:var(--t1);outline:none;transition:border-color .12s}
// // .tag-add-input:focus{border-color:rgba(99,102,241,.4)}
// // .tag-add-btn{background:var(--bg3);border:1px solid var(--b1);border-radius:6px;padding:5px 8px;cursor:pointer;color:var(--t3);display:flex;transition:all .12s}
// // .tag-add-btn:hover{border-color:rgba(99,102,241,.3);color:var(--ac-l)}

// // /* ═══ TRASH ═══════════════════════════════════════════════════════════════ */
// // .trash-card{position:relative;aspect-ratio:1;border-radius:var(--r2);overflow:hidden;background:var(--bg2);border:1px solid var(--b1)}
// // .trash-img{width:100%;height:100%;object-fit:cover;filter:grayscale(.5) brightness(.4)}
// // .trash-overlay{position:absolute;inset:0;opacity:0;transition:opacity .18s;background:rgba(0,0,0,.7);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:6px;padding:10px}
// // .trash-card:hover .trash-overlay{opacity:1}
// // .trash-btn{width:100%;display:flex;align-items:center;justify-content:center;gap:5px;border:none;border-radius:var(--r);padding:7px;font-family:var(--font);font-size:11px;font-weight:600;cursor:pointer;transition:all .12s}
// // .trash-btn--restore{background:var(--ac);color:#fff}
// // .trash-btn--del{background:rgba(220,50,50,.85);color:#fff}
// // .trash-date{position:absolute;top:7px;left:7px;font-family:var(--mono);font-size:7.5px;background:rgba(0,0,0,.7);color:var(--t3);padding:2px 6px;border-radius:4px}

// // /* ═══ PEOPLE ══════════════════════════════════════════════════════════════ */
// // .people-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(108px,1fr));gap:18px}
// // .person-card{display:flex;flex-direction:column;align-items:center;gap:8px;cursor:pointer}
// // .person-avatar{width:68px;height:68px;border-radius:50%;overflow:hidden;border:1px solid var(--b1);background:var(--bg2);display:flex;align-items:center;justify-content:center;transition:border-color .15s}
// // .person-avatar img{width:100%;height:100%;object-fit:cover}
// // .person-card:hover .person-avatar{border-color:var(--ac)}
// // .person-name{font-size:12px;font-weight:600;text-align:center}
// // .person-count{font-family:var(--mono);font-size:8px;color:var(--t3)}

// // /* ═══ ALBUMS ══════════════════════════════════════════════════════════════ */
// // .albums-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px}
// // .album-card{
// //   position:relative;aspect-ratio:16/10;
// //   border-radius:var(--r3);overflow:hidden;cursor:pointer;
// //   background:var(--bg2);border:1px solid var(--b1);
// //   transition:border-color .15s;
// // }
// // .album-card:hover{border-color:var(--b2)}
// // .album-cover{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;transition:transform .3s}
// // .album-card:hover .album-cover{transform:scale(1.04)}
// // .album-cover-ph{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;color:var(--t3)}
// // .album-strip{position:absolute;bottom:0;left:0;right:0;height:28px;display:flex;opacity:0;transition:opacity .18s;z-index:2}
// // .album-card:hover .album-strip{opacity:1}
// // .album-strip-img{flex:1;height:100%;object-fit:cover;border-right:1px solid rgba(0,0,0,.3)}
// // .album-info{position:absolute;inset:0;z-index:1;background:linear-gradient(to top,rgba(0,0,0,.9) 0%,transparent 55%);padding:12px;display:flex;flex-direction:column;justify-content:flex-end}
// // .album-title{font-weight:700;font-size:13px;letter-spacing:-.01em;margin-bottom:5px;line-height:1.3;color:#fff}
// // .album-meta{display:flex;align-items:center;gap:5px;flex-wrap:wrap}
// // .album-del-btn{position:absolute;top:8px;right:8px;width:24px;height:24px;border-radius:50%;background:rgba(0,0,0,.65);border:1px solid var(--b2);color:var(--t3);display:flex;align-items:center;justify-content:center;cursor:pointer;z-index:3;opacity:0;transition:all .15s}
// // .album-card:hover .album-del-btn{opacity:1}
// // .album-del-btn:hover{background:rgba(220,50,50,.85);color:#fff;border-color:transparent}
// // .tag-sm{font-family:var(--mono);font-size:8px;background:rgba(255,255,255,.08);color:var(--t3);padding:2px 6px;border-radius:20px}
// // .tag-sm--accent{background:rgba(99,102,241,.15);color:var(--ac-l)}
// // .tag-sm--manual{background:rgba(245,158,11,.1);color:rgba(245,158,11,.8)}

// // /* ═══ DUPLICATES ══════════════════════════════════════════════════════════ */
// // .dupe-list{display:flex;flex-direction:column;gap:14px}
// // .dupe-group{background:var(--bg2);border:1px solid var(--b1);border-radius:var(--r3);padding:16px}
// // .dupe-head{display:flex;align-items:center;gap:7px;margin-bottom:14px;flex-wrap:wrap}

// // /* ═══ MODAL ═══════════════════════════════════════════════════════════════ */
// // .modal-bg{position:fixed;inset:0;z-index:200;background:rgba(0,0,0,.8);backdrop-filter:blur(10px);display:flex;align-items:center;justify-content:center;padding:18px}
// // .modal-box{width:100%;max-height:90vh;overflow-y:auto;background:var(--bg1);border:1px solid var(--b1);border-radius:var(--r3);padding:20px;box-shadow:0 16px 50px rgba(0,0,0,.7)}
// // .modal-head{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:14px;gap:10px}
// // .modal-title{font-size:18px;font-weight:700;letter-spacing:-.025em}
// // .modal-sub{font-family:var(--mono);font-size:8px;color:var(--t3);letter-spacing:.06em;margin-top:3px}
// // .rename-row{display:flex;gap:7px;margin-bottom:14px;align-items:center}
// // .rename-in{flex:1;background:var(--bg2);border:1px solid var(--b1);border-radius:var(--r);padding:8px 11px;font-family:var(--font);font-size:13px;color:var(--t1);outline:none;transition:border-color .12s}
// // .rename-in:focus{border-color:rgba(99,102,241,.4)}
// // .album-desc{background:var(--ac-bg);border:1px solid rgba(99,102,241,.15);border-radius:var(--r);padding:10px 12px;margin-bottom:14px}
// // .album-desc-lbl{font-family:var(--mono);font-size:8px;color:var(--ac-l);letter-spacing:.08em;display:block;margin-bottom:4px}
// // .album-desc p{font-size:12px;line-height:1.6}
// // .create-album-form{display:flex;flex-direction:column;gap:9px}
// // .form-input{background:var(--bg2);border:1px solid var(--b1);border-radius:var(--r);padding:10px 12px;font-family:var(--font);font-size:13px;color:var(--t1);outline:none;transition:border-color .12s}
// // .form-input:focus{border-color:rgba(99,102,241,.4)}
// // .form-textarea{background:var(--bg2);border:1px solid var(--b1);border-radius:var(--r);padding:10px 12px;font-family:var(--font);font-size:13px;color:var(--t1);outline:none;resize:vertical;transition:border-color .12s}
// // .form-textarea:focus{border-color:rgba(99,102,241,.4)}

// // .btn-sm{display:inline-flex;align-items:center;gap:5px;background:var(--bg2);border:1px solid var(--b1);color:var(--t2);border-radius:var(--r);padding:6px 12px;font-family:var(--font);font-size:11px;font-weight:500;cursor:pointer;transition:all .12s}
// // .btn-sm:hover{border-color:var(--b2);color:var(--t1);background:var(--bg3)}
// // .btn-sm--primary{background:var(--ac);border-color:transparent;color:#fff}
// // .btn-sm--primary:hover{background:var(--ac-d)}
// // .btn-sm--accent{background:var(--ac-bg);border-color:rgba(99,102,241,.25);color:var(--ac-l)}
// // .btn-sm--accent:hover{background:var(--ac);color:#fff;border-color:transparent}
// // .btn-sm:disabled{opacity:.35;cursor:default}

// // /* ═══ STATS ═══════════════════════════════════════════════════════════════ */
// // .stats-wrap{max-width:820px}
// // .stats-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:14px}
// // .stat-card{
// //   background:var(--bg2);border:1px solid var(--b1);
// //   border-radius:var(--r2);padding:16px 18px;
// //   border-top:2px solid var(--sc);
// //   transition:border-color .15s;
// // }
// // .stat-card:hover{border-color:var(--b2)}
// // .stat-lbl{font-family:var(--mono);font-size:8px;color:var(--t3);letter-spacing:.09em;margin-bottom:8px;text-transform:uppercase}
// // .stat-val{font-size:36px;font-weight:800;letter-spacing:-.04em;line-height:1;color:var(--sc)}
// // .stats-sec{background:var(--bg2);border:1px solid var(--b1);border-radius:var(--r2);padding:14px;margin-bottom:8px}
// // .stats-sec-lbl{font-family:var(--mono);font-size:8px;color:var(--t3);letter-spacing:.09em;margin-bottom:12px;text-transform:uppercase}
// // .clr-dist{display:flex;flex-wrap:wrap;gap:6px}
// // .clr-dist-item{display:flex;align-items:center;gap:6px;background:var(--bg3);border:1px solid var(--b1);border-radius:6px;padding:5px 10px;font-size:11px}
// // .clr-dist-dot{width:10px;height:10px;border-radius:50%;border:1px solid rgba(255,255,255,.1);flex-shrink:0}
// // .clr-dist-n{font-family:var(--mono);font-size:9px;color:var(--t3)}
// // .tag-cloud{display:flex;flex-wrap:wrap;gap:5px}
// // .tag-cloud-item{font-family:var(--mono);font-size:9px;background:var(--ac-bg);color:var(--ac-l);border:1px solid rgba(99,102,241,.15);padding:3px 9px;border-radius:20px}
// // .tag-n{opacity:.5}

// // /* ═══ EMPTY / LOADER ══════════════════════════════════════════════════════ */
// // .empty{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:240px;background:var(--bg2);border:1px dashed var(--b1);border-radius:var(--r3);padding:40px;gap:10px;text-align:center}
// // .empty--click{cursor:pointer;transition:background .12s}
// // .empty--click:hover{background:var(--bg3)}
// // .empty-msg{font-size:16px;font-weight:700;letter-spacing:-.02em}
// // .empty-sub{font-size:12px;color:var(--t2);max-width:240px;line-height:1.6}
// // .empty-cta{font-family:var(--mono);font-size:9px;color:var(--ac-l);letter-spacing:.09em;margin-top:4px;text-transform:uppercase}
// // .loader-wrap{display:flex;flex-direction:column;align-items:center;justify-content:center;height:55vh;gap:12px}
// // .loader-ring{width:34px;height:34px;border-radius:50%;border:2px solid rgba(99,102,241,.12);border-top-color:var(--ac);animation:spin .7s linear infinite}
// // .loader-text{font-family:var(--mono);font-size:10px;color:var(--t3);letter-spacing:.1em}
// // @keyframes spin{to{transform:rotate(360deg)}}
// // .spin{animation:spin .7s linear infinite}

// // /* ═══ MISC ════════════════════════════════════════════════════════════════ */
// // .person-chip{display:flex;align-items:center;padding:5px 11px;background:var(--bg2);border:1px solid var(--b1);border-radius:20px;font-size:12px;color:var(--t3);cursor:pointer;transition:all .12s}
// // .person-chip:hover{border-color:rgba(99,102,241,.3);color:var(--t2)}
// // .person-chip--on{background:rgba(49,46,129,.4);border-color:var(--ac);color:var(--ac-l)}

// // @media(max-width:720px){
// //   .lb-box{grid-template-columns:1fr}
// //   .lb-right{border-left:none;border-top:1px solid var(--b1);max-height:40vh}
// //   .stats-cards{grid-template-columns:repeat(2,1fr)}
// //   .sidebar{width:180px}
// // }
// // `}</style>;
// // }

// import React, { useState, useEffect, useRef, useCallback } from "react";
// import { motion, AnimatePresence } from "framer-motion";
// import {
//   Search, Upload, Image as ImgIcon, Clock, BookOpen, Users, Heart,
//   Copy, Trash2, X, ChevronLeft, ChevronRight, Sparkles, BarChart3,
//   GitMerge, Palette, Shuffle, RotateCcw, AlertTriangle, FileText,
//   Mic, MicOff, Zap, Star, Eye, Camera, Info, Tag, Type, Smile,
//   Loader2, Trash, BookImage, ImageOff, TrendingUp, Aperture,
//   CheckSquare, Square, Plus, Calendar, Hash, FolderPlus,
//   CheckCheck, Layers, ChevronDown, ChevronUp,
//   SlidersHorizontal, UserCheck
// } from "lucide-react";
// import axios from "axios";

// const API = "http://localhost:8000";
// const imgUrl = (f) => {
//   if (!f) return null;
//   const b = String(f).split("/").filter(Boolean).pop();
//   return b ? `${API}/image/${b}` : null;
// };
// const BLANK = `data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='1' height='1'%3E%3Crect fill='%23111118'/%3E%3C/svg%3E`;

// const QL = {
//   Excellent: { c:"#4ade80", bg:"rgba(74,222,128,.12)" },
//   Good:      { c:"#818cf8", bg:"rgba(129,140,248,.12)" },
//   Fair:      { c:"#fbbf24", bg:"rgba(251,191,36,.12)"  },
//   Poor:      { c:"#f87171", bg:"rgba(248,113,113,.12)" },
// };
// const EMO = { happy:"😊",sad:"😢",angry:"😠",surprised:"😲",disgusted:"🤢",fearful:"😨",neutral:"😐" };
// const EMO_COLORS = { happy:"#fbbf24",sad:"#60a5fa",angry:"#f87171",surprised:"#a78bfa",disgusted:"#4ade80",fearful:"#fb923c",neutral:"#6b7280" };
// const qInfo = (l) => QL[l] || { c:"#555", bg:"rgba(255,255,255,.05)" };

// /* ─── Voice hook ─────────────────────────────────────────────────────────── */
// function useVoice(onResult, onError) {
//   const [state, setState] = useState("idle");
//   const streamRef = useRef(null);
//   const processorRef = useRef(null);
//   const ctxRef = useRef(null);
//   const samplesRef = useRef([]);
//   const ok = typeof navigator !== "undefined" && !!navigator.mediaDevices?.getUserMedia;

//   // Convert Float32 PCM samples → 16-bit WAV blob (no ffmpeg needed)
//   const buildWav = (samples, sampleRate) => {
//     const len = samples.length;
//     const buf = new ArrayBuffer(44 + len * 2);
//     const view = new DataView(buf);
//     const write = (off, str) => { for(let i=0;i<str.length;i++) view.setUint8(off+i, str.charCodeAt(i)); };
//     write(0,"RIFF"); view.setUint32(4, 36+len*2, true);
//     write(8,"WAVE"); write(12,"fmt ");
//     view.setUint32(16,16,true); view.setUint16(20,1,true); view.setUint16(22,1,true);
//     view.setUint32(24,sampleRate,true); view.setUint32(28,sampleRate*2,true);
//     view.setUint16(32,2,true); view.setUint16(34,16,true);
//     write(36,"data"); view.setUint32(40,len*2,true);
//     for(let i=0;i<len;i++){
//       const s = Math.max(-1,Math.min(1,samples[i]));
//       view.setInt16(44+i*2, s<0?s*0x8000:s*0x7FFF, true);
//     }
//     return new Blob([buf], {type:"audio/wav"});
//   };

//   const start = useCallback(async () => {
//     if (!ok) return onError?.("Mic unavailable");
//     try {
//       const stream = await navigator.mediaDevices.getUserMedia({
//         audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true }
//       });
//       streamRef.current = stream;
//       samplesRef.current = [];

//       // Use Web Audio API to capture raw PCM at 16kHz — bypasses ffmpeg completely
//       const ctx = new AudioContext({ sampleRate: 16000 });
//       ctxRef.current = ctx;
//       const source = ctx.createMediaStreamSource(stream);
//       const processor = ctx.createScriptProcessor(4096, 1, 1);
//       processorRef.current = processor;

//       processor.onaudioprocess = e => {
//         const data = e.inputBuffer.getChannelData(0);
//         samplesRef.current.push(new Float32Array(data));
//       };
//       source.connect(processor);
//       processor.connect(ctx.destination);
//       setState("rec");
//     } catch(e) {
//       setState("error");
//       onError?.(e.name === "NotAllowedError" ? "Mic permission denied" : "Mic unavailable");
//       setTimeout(()=>setState("idle"),3000);
//     }
//   }, [ok]);

//   const stop = useCallback(async () => {
//     if (state !== "rec") return;
//     setState("processing");
//     try {
//       // Stop all tracks
//       streamRef.current?.getTracks().forEach(t=>t.stop());
//       processorRef.current?.disconnect();
//       await ctxRef.current?.close();

//       // Flatten all captured chunks into one array
//       const allSamples = samplesRef.current;
//       if (!allSamples.length) {
//         setState("error"); onError?.("No audio captured"); setTimeout(()=>setState("idle"),3000); return;
//       }
//       const total = allSamples.reduce((s,c)=>s+c.length,0);
//       const merged = new Float32Array(total);
//       let off=0; for(const c of allSamples){merged.set(c,off);off+=c.length;}

//       // Build WAV and send to Vosk (no ffmpeg needed — already 16kHz mono PCM)
//       const wav = buildWav(merged, 16000);
//       const fd = new FormData(); fd.append("audio", wav, "rec.wav");
//       const {data} = await axios.post(`${API}/voice_search`, fd, {timeout:30000});
//       if (data.success && data.transcript) { setState("idle"); onResult(data.transcript); }
//       else { setState("error"); onError?.(data.error||"No speech detected — speak clearly and try again"); setTimeout(()=>setState("idle"),4000); }
//     } catch(err) {
//       const detail = err?.response?.data?.detail || err.message || "";
//       setState("error");
//       onError?.(err?.response?.status===503 ? "Vosk model not loaded" : detail.slice(0,60) || "Voice failed");
//       setTimeout(()=>setState("idle"),5000);
//     }
//   }, [state]);

//   return { state, ok, start, stop };
// }

// /* ══════════════════════════════════════════════════════════════════════════ */
// export default function App() {
//   const [view,        setView]        = useState("timeline");
//   const [images,      setImages]      = useState([]);
//   const [faces,       setFaces]       = useState([]);
//   const [albums,      setAlbums]      = useState([]);
//   const [dupes,       setDupes]       = useState([]);
//   const [trash,       setTrash]       = useState([]);
//   const [stats,       setStats]       = useState(null);
//   const [loading,     setLoading]     = useState(false);
//   const [searchQ,     setSearchQ]     = useState("");
//   const [searchMode,  setSearchMode]  = useState("text");
//   const [colorPick,   setColorPick]   = useState("blue");
//   const [hybridFile,  setHybridFile]  = useState(null);
//   const [hybridPrev,  setHybridPrev]  = useState(null);
//   const [voiceErr,    setVoiceErr]    = useState("");
//   const [reindexMsg,  setReindexMsg]  = useState("");
//   const [reindexing,  setReindexing]  = useState(false);

//   /* New feature state */
//   const [emoSummary,    setEmoSummary]    = useState([]);
//   const [emoFilter,     setEmoFilter]     = useState("");        // active emotion filter
//   const [onThisDay,     setOnThisDay]     = useState(null);
//   const [otdExpanded,   setOtdExpanded]   = useState(false);

//   const [batchMode,     setBatchMode]     = useState(false);
//   const [selected,      setSelected]      = useState(new Set());
//   const [batchTagInput, setBatchTagInput] = useState("");
//   const [batchAlbumId,  setBatchAlbumId]  = useState("");

//   const [allTags,       setAllTags]       = useState([]);
//   const [activeTag,     setActiveTag]     = useState("");

//   const [createAlbumOpen, setCreateAlbumOpen] = useState(false);
//   const [newAlbumTitle,   setNewAlbumTitle]   = useState("");
//   const [newAlbumDesc,    setNewAlbumDesc]    = useState("");
//   const [pendingAlbumImages, setPendingAlbumImages] = useState([]); // snapshot of selected at open time

//   // Named event creation
//   const [createEventOpen, setCreateEventOpen] = useState(false);
//   const [newEventTitle,   setNewEventTitle]   = useState("");
//   const [newEventType,    setNewEventType]    = useState("Other");
//   const [newEventDate,    setNewEventDate]    = useState("");
//   const [newEventDesc,    setNewEventDesc]    = useState("");

//   // Advanced filters
//   const [minPeople,     setMinPeople]     = useState(0);
//   const [coPersonIds,   setCoPersonIds]   = useState([]);
//   const [showCoModal,   setShowCoModal]   = useState(false);

//   // Face similarity
//   const [faceSrcFile,   setFaceSrcFile]   = useState(null);
//   const [faceSrcPrev,   setFaceSrcPrev]   = useState(null);
//   const [showFaceSim,   setShowFaceSim]   = useState(false);

//   // People frequency
//   const [peopleFreq,    setPeopleFreq]    = useState([]);
//   // People suggestions shown when search returns no results
//   const [peopleSuggestions, setPeopleSuggestions] = useState([]);

//   /* Lightbox */
//   const [lb, setLb] = useState(null);
//   /* Modals */
//   const [personModal, setPersonModal] = useState(null);
//   const [albumModal,  setAlbumModal]  = useState(null);

//   const uploadRef = useRef(null);
//   const voice = useVoice(
//     t => { setSearchQ(t); setSearchMode("text"); setVoiceErr(""); runSearch("text",t); },
//     setVoiceErr
//   );

//   /* keyboard */
//   useEffect(() => {
//     const h = e => {
//       if (!lb) return;
//       if (e.key==="ArrowRight") shiftLb(1);
//       if (e.key==="ArrowLeft")  shiftLb(-1);
//       if (e.key==="Escape")     setLb(null);
//     };
//     window.addEventListener("keydown",h);
//     return ()=>window.removeEventListener("keydown",h);
//   }, [lb]);

//   const shiftLb = useCallback(d=>setLb(p=>p?{...p,idx:(p.idx+d+p.list.length)%p.list.length}:null),[]);
//   const openLb  = (list,idx)=>setLb({list,idx});
//   const closeLb = ()=>setLb(null);

//   /* load */
//   useEffect(()=>{ load(); },[view]);
//   useEffect(()=>{
//     loadEmoSummary(); loadTags(); loadTrashCount(); loadPeopleFreq();

//   }, []);
//   useEffect(()=>{ if(view==="timeline") loadOnThisDay(); }, [view]);

//   const load = async () => {
//     setLoading(true);
//     try {
//       if (view==="timeline")   { const r=await axios.get(`${API}/timeline`);              setImages(r.data.results||[]); }
//       if (view==="favorites")  { const r=await axios.get(`${API}/favorites`);             setImages(r.data.results||[]); }
//       if (view==="explore")    { const r=await axios.get(`${API}/explore/random?count=24`);setImages(r.data.results||[]); }
//       if (view==="faces")      { const r=await axios.get(`${API}/faces`);                 setFaces(r.data.results||[]); }
//       if (view==="albums")     { const r=await axios.get(`${API}/albums`);                setAlbums(r.data.results||[]); }
//       if (view==="duplicates") { const r=await axios.get(`${API}/duplicates`);            setDupes(r.data.duplicate_groups||[]); }
//       if (view==="trash")      { const r=await axios.get(`${API}/trash`);                 setTrash(r.data.results||[]); }
//       if (view==="stats")      { const r=await axios.get(`${API}/stats`);                 setStats(r.data); }
//     } catch {}
//     setSelected(new Set()); setBatchMode(false);
//     setLoading(false);
//   };

//   const loadEmoSummary = async () => {
//     try { const r=await axios.get(`${API}/emotions/summary`); setEmoSummary(r.data.emotions||[]); } catch {}
//   };
//   const loadOnThisDay = async () => {
//     try { const r=await axios.get(`${API}/on-this-day`); setOnThisDay(r.data); } catch {}
//   };
//   const loadTags = async () => {
//     try { const r=await axios.get(`${API}/tags`); setAllTags(r.data.tags||[]); } catch {}
//   };
//   const loadTrashCount = async () => {
//     try { const r=await axios.get(`${API}/trash`); setTrash(r.data.results||[]); } catch {}
//   };

//   const loadPeopleFreq = async () => {
//     try { const r=await axios.get(`${API}/people/frequency`); setPeopleFreq(r.data.people||[]); } catch {}
//   };

//   // Group photo filter
//   const filterGroupPhotos = async (n) => {
//     setMinPeople(n); setLoading(true); setView("search");
//     try { const r=await axios.get(`${API}/group-photos?min_people=${n}`); setImages(r.data.results||[]); }
//     catch {} finally { setLoading(false); }
//   };

//   // Co-occurrence search
//   const runCoOccurrence = async (ids) => {
//     if (!ids||ids.length<2) return;
//     setLoading(true); setView("search"); setShowCoModal(false);
//     try { const r=await axios.get(`${API}/co-occurrence?person_ids=${ids.join(",")}`); setImages(r.data.results||[]); }
//     catch {} finally { setLoading(false); }
//   };

//   // Face similarity
//   const runFaceSimilarity = async () => {
//     if (!faceSrcFile) return;
//     setLoading(true); setView("search"); setShowFaceSim(false);
//     const fd=new FormData(); fd.append("file", faceSrcFile); fd.append("top_k","30");
//     try { const r=await axios.post(`${API}/face-similarity`, fd); setImages(r.data.results||[]); }
//     catch(err) { alert("Face search failed: "+(err?.response?.data?.detail||err.message)); }
//     finally { setLoading(false); setFaceSrcFile(null); setFaceSrcPrev(null); }
//   };

//   const runSearch = async (mode=searchMode, q=searchQ) => {
//     setEmoFilter(""); setActiveTag(""); setLoading(true); setView("search");
//     try {
//       const fd=new FormData(); let res;
//       if (mode==="text")     { fd.append("query",q);          res=await axios.post(`${API}/search`,fd); }
//       if (mode==="describe") { fd.append("description",q);    res=await axios.post(`${API}/search/describe`,fd); }
//       if (mode==="image")    { fd.append("file",hybridFile); fd.append("top_k",20); res=await axios.post(`${API}/search/image`,fd); }
//       if (mode==="hybrid")   { fd.append("query",q); if(hybridFile)fd.append("file",hybridFile); fd.append("top_k",20); res=await axios.post(`${API}/search/hybrid`,fd); }
//       if (mode==="color")    { fd.append("color",colorPick); fd.append("top_k",20); res=await axios.post(`${API}/search/color`,fd); }
//       let results = res?.data?.results || [];
//       // If text search returned nothing, try person name search as fallback
//       // Only for queries that look like person names (not animals/objects/descriptive)
//       const _NON_PERSON = new Set([
//         "cat","dog","horse","cow","bird","fish","fox","lion","tiger","bear","wolf","pig",
//         "rabbit","duck","frog","snake","deer","sheep","goat","chicken","otter","seal",
//         "panda","koala","monkey","giraffe","zebra","elephant","whale","shark","eagle",
//         "owl","penguin","parrot","bee","ant","spider","butterfly","kitten","puppy",
//         "sunset","beach","ocean","mountain","forest","river","flower","tree","sky",
//         "snow","rain","night","city","park","road","bridge","building","house","lake",
//         "football","soccer","cricket","basketball","tennis","volleyball","swimming",
//         "running","cycling","boxing","golf","hockey","baseball","rugby","sport","sports",
//         "game","match","player","athlete","stadium","gym","yoga","dance","exercise",
//         "car","bus","train","plane","boat","ship","bike","truck",
//         "food","pizza","cake","coffee","burger","sushi","pasta","bread",
//         "sunrise","rainbow","storm","cloud","wind","landscape","nature","sky",
//       ]);
//       const _qWords = q.trim().toLowerCase().split(/\s+/);
//       const _isPersonQuery = _qWords.length <= 4 &&
//         _qWords.some(w => w.length >= 3 && !_NON_PERSON.has(w)) &&
//         !_qWords.every(w => ["a","an","the","in","on","at","of","for","with",
//           "woman","man","girl","boy","black","white","red","blue","green","yellow",
//           "dress","shirt","jacket","hair","wearing","standing"].includes(w));
//       if (mode==="text" && results.length===0 && q.trim().length>=2 && _isPersonQuery) {
//         try {
//           const pr = await axios.get(`${API}/people/search?q=${encodeURIComponent(q.trim())}`);
//           if (pr.data.results?.length) {
//             results = pr.data.results.flatMap(p => p.results || []);
//           }
//         } catch {}
//       }
//       // Show people suggestions if still no results
//       if (mode==="text" && results.length===0) {
//         const suggestions = res?.data?.people_suggestions || [];
//         setPeopleSuggestions(suggestions);
//       } else {
//         setPeopleSuggestions([]);
//       }
//       setImages(results);
//     } catch { setImages([]); }
//     setLoading(false);
//   };

//   const filterByEmotion = async (emo) => {
//     if (emoFilter===emo) { setEmoFilter(""); load(); return; }
//     setEmoFilter(emo); setView("search"); setLoading(true);
//     try { const fd=new FormData(); fd.append("emotion",emo); const r=await axios.post(`${API}/search/emotion`,fd); setImages(r.data.results||[]); }
//     catch { setImages([]); }
//     setLoading(false);
//   };

//   const filterByTag = async (tag) => {
//     if (activeTag===tag) { setActiveTag(""); load(); return; }
//     setActiveTag(tag); setView("timeline"); setLoading(true);
//     try { const r=await axios.get(`${API}/tags/${tag}/images`); setImages(r.data.results||[]); }
//     catch { setImages([]); }
//     setLoading(false);
//   };

//   const handleUpload = async (files) => {
//     setLoading(true);
//     for (const f of Array.from(files)) { try { const fd=new FormData(); fd.append("file",f); await axios.post(`${API}/upload`,fd); } catch {} }
//     load();
//   };

//   const doReindex = async () => {
//     setReindexing(true); setReindexMsg("Clustering…");
//     try { const r=await axios.post(`${API}/recluster`); setReindexMsg(`✓ ${r.data.people} people · ${r.data.albums} albums`); setTimeout(load,500); }
//     catch { setReindexMsg("Failed"); }
//     setReindexing(false);
//   };

//   const reprocessEmotions = async () => {
//     setReindexing(true); setReindexMsg("Reprocessing emotions…");
//     try {
//       const r = await axios.post(`${API}/reprocess-emotions`);
//       setReindexMsg(`✓ ${r.data.message}`);
//       setTimeout(()=>{ loadEmoSummary(); if(view==="stats") load(); }, 3000);
//     } catch(err) {
//       setReindexMsg("Emotion reprocess failed: " + (err?.response?.data?.detail || err.message));
//     }
//     setReindexing(false);
//   };

//   const reprocessColors = async () => {
//     setReindexing(true); setReindexMsg("Recomputing colors…");
//     try {
//       const r = await axios.post(`${API}/reprocess-colors`);
//       setReindexMsg(`✓ Colors updated for ${r.data.updated} photos`);
//     } catch(err) {
//       setReindexMsg("Color reprocess failed: " + (err?.response?.data?.detail || err.message));
//     }
//     setReindexing(false);
//   };

//   const reprocessNames = async () => {
//     setReindexing(true); setReindexMsg("Auto-naming people…");
//     try {
//       const r = await axios.post(`${API}/reprocess-names`);
//       setReindexMsg(`✓ ${r.data.message}`);
//       setTimeout(()=>{ if(view==="faces") load(); }, 1000);
//     } catch(err) {
//       setReindexMsg("Name detection failed: " + (err?.response?.data?.detail || err.message));
//     }
//     setReindexing(false);
//   };

//   const reprocessCaptions = async (forceAll=false) => {
//     setReindexing(true); setReindexMsg(forceAll ? "Re-captioning all photos…" : "Captioning photos missing descriptions…");
//     try {
//       const r = await axios.post(`${API}/recaption${forceAll ? "?force_all=true" : ""}`);
//       if (r.data.status === "nothing_to_do") {
//         setReindexMsg("✓ All photos already have captions. Use 'Re-caption All' to force.");
//       } else {
//         setReindexMsg(`✓ ${r.data.message}`);
//       }
//     } catch(err) {
//       setReindexMsg("Caption failed: " + (err?.response?.data?.detail || err.message));
//     }
//     setReindexing(false);
//   };

//   const toggleFav = async (id, e) => {
//     e?.stopPropagation();
//     // Optimistic update: flip is_favorite in local state immediately so heart turns red/grey right away
//     setImages(prev => prev.map(img => img.id === id ? {...img, is_favorite: !img.is_favorite} : img));
//     try {
//       const fd=new FormData();
//       fd.append("image_id", id);
//       const res = await axios.post(`${API}/toggle_favorite`, fd);
//       // Sync with server response to make sure we're in the right state
//       if (res.data && res.data.is_favorite !== undefined) {
//         setImages(prev => prev.map(img => img.id === id ? {...img, is_favorite: res.data.is_favorite} : img));
//       }
//       if (view==="favorites") load();
//     } catch {
//       // Revert optimistic update on error
//       setImages(prev => prev.map(img => img.id === id ? {...img, is_favorite: !img.is_favorite} : img));
//     }
//   };

//   const softDelete = async (id, e) => {
//     e?.stopPropagation();
//     if (!window.confirm("Move to trash?")) return;
//     try {
//       const fd=new FormData(); fd.append("image_id",id);
//       await axios.post(`${API}/delete_image`,fd);
//       setImages(p=>p.filter(i=>i.id!==id));
//       loadTrashCount();
//     } catch(err) {
//       alert("Could not move to trash: " + (err?.response?.data?.detail || err.message));
//     }
//   };

//   const restoreImg = async id => {
//     try {
//       const fd=new FormData(); fd.append("image_id",id);
//       await axios.post(`${API}/restore`,fd);
//       setTrash(p=>p.filter(i=>i.id!==id));
//     } catch(err) {
//       alert("Restore failed: " + (err?.response?.data?.detail || err.message));
//     }
//   };
//   const permDelete = async id => {
//     if (!window.confirm("Delete forever? This cannot be undone.")) return;
//     try {
//       const fd=new FormData(); fd.append("image_id",id);
//       await axios.post(`${API}/permanent_delete`,fd);
//       setTrash(p=>p.filter(i=>i.id!==id));
//     } catch(err) {
//       alert("Delete failed: " + (err?.response?.data?.detail || err.message));
//     }
//   };

//   const openPerson = async p => {
//     setPersonModal({...p,images:null});
//     try { const r=await axios.get(`${API}/people/${p.id}`); setPersonModal(r.data); } catch {}
//   };
//   const openAlbum = async a => {
//     setAlbumModal({...a,images:null});
//     try { const r=await axios.get(`${API}/albums/${a.id}`); setAlbumModal(r.data); } catch {}
//   };
//   const renamePerson = async (id,name) => {
//     try { const fd=new FormData(); fd.append("name",name); await axios.post(`${API}/people/${id}`,fd); load(); setPersonModal(p=>p&&{...p,name}); } catch {}
//   };

//   /* Batch actions */
//   const toggleSelect = useCallback((id) => {
//     setSelected(prev => { const s=new Set(prev); s.has(id)?s.delete(id):s.add(id); return s; });
//   }, []);
//   const selectAll = () => setSelected(new Set(images.map(i=>i.id)));
//   const clearSel  = () => setSelected(new Set());

//   const batchFavorite = async (val) => {
//     if (!selected.size) return;
//     const fd=new FormData(); fd.append("image_ids",[...selected].join(",")); fd.append("value",val?1:0);
//     try { await axios.post(`${API}/batch/favorite`,fd); load(); } catch {}
//   };
//   const batchDelete = async () => {
//     if (!selected.size||!window.confirm(`Trash ${selected.size} photos?`)) return;
//     const fd=new FormData(); fd.append("image_ids",[...selected].join(","));
//     try { await axios.post(`${API}/batch/delete`,fd); load(); } catch {}
//   };
//   const batchTag = async () => {
//     if (!selected.size||!batchTagInput.trim()) return;
//     const fd=new FormData(); fd.append("image_ids",[...selected].join(",")); fd.append("tag",batchTagInput.trim());
//     try { await axios.post(`${API}/batch/tag`,fd); setBatchTagInput(""); loadTags(); } catch {}
//   };
//   const batchAddAlbum = async () => {
//     if (!selected.size||!batchAlbumId) return;
//     const fd=new FormData(); fd.append("image_ids",[...selected].join(",")); fd.append("album_id",batchAlbumId);
//     try { await axios.post(`${API}/batch/album`,fd); setBatchAlbumId(""); load(); } catch {}
//   };

//   const createAlbum = async (imageIds=[]) => {
//     if (!newAlbumTitle.trim()) return;
//     const fd=new FormData();
//     fd.append("title", newAlbumTitle);
//     fd.append("description", newAlbumDesc);
//     // Priority: explicit imageIds arg → pendingAlbumImages snapshot → current selected
//     const ids = imageIds.length ? imageIds
//               : pendingAlbumImages.length ? pendingAlbumImages
//               : [...selected];
//     console.log("Creating album with", ids.length, "images:", ids);
//     if (ids.length) fd.append("image_ids", ids.join(","));
//     try {
//       const r = await axios.post(`${API}/albums/create`, fd);
//       setCreateAlbumOpen(false); setNewAlbumTitle(""); setNewAlbumDesc("");
//       const addedCount = ids.length;
//       setPendingAlbumImages([]);
//       setSelected(new Set()); setBatchMode(false);
//       setView("albums");
//       await load();
//       // Open the newly created album immediately so user can see it
//       if (r.data?.id) {
//         const newAlbum = {id: r.data.id, title: r.data.title, type:"manual",
//                          count: addedCount, images: null};
//         openAlbum(newAlbum);
//       }
//       return r.data;
//     } catch(err) { alert("Failed to create album: " + (err?.response?.data?.detail || err.message)); }
//   };
//   const deleteAlbum = async id => {
//     try {
//       await axios.delete(`${API}/albums/${id}/delete`);
//       setAlbumModal(null);
//       load();
//     } catch(err) { alert("Delete failed: " + (err?.response?.data?.detail || err.message)); }
//   };

//   const renameAlbum = async (id, title, description) => {
//     const fd=new FormData(); fd.append("title", title);
//     if (description) fd.append("description", description);
//     try {
//       await axios.post(`${API}/albums/${id}/rename`, fd);
//       load();
//       setAlbumModal(prev => prev && prev.id===id ? {...prev, title, description: description||prev.description} : prev);
//     } catch(err) { alert("Rename failed: " + (err?.response?.data?.detail || err.message)); }
//   };

//   const EVENT_TYPES = ["Birthday","Trip","Vacation","Wedding","Anniversary","Graduation","Party","Holiday","Family","Work","Other"];

//   const createNamedEvent = async () => {
//     if (!newEventTitle.trim()) return;
//     const fd=new FormData();
//     fd.append("title", newEventTitle.trim());
//     fd.append("event_type", newEventType);
//     fd.append("description", newEventDesc.trim());
//     fd.append("date_str", newEventDate);
//     try {
//       await axios.post(`${API}/events/create`, fd);
//       setCreateEventOpen(false);
//       setNewEventTitle(""); setNewEventType("Other"); setNewEventDate(""); setNewEventDesc("");
//       load();
//     } catch(err) { alert("Could not create event: " + (err?.response?.data?.detail || err.message)); }
//   };

//   const addTagToImage = async (imgId, tag) => {
//     const fd=new FormData(); fd.append("tag",tag);
//     try {
//       const r = await axios.post(`${API}/photo/${imgId}/tags/add`,fd);
//       const newTags = r.data.tags || [];
//       // Update the images list in place so lightbox reflects latest tags
//       setImages(prev => prev.map(i => i.id===imgId ? {...i, user_tags: newTags} : i));
//       loadTags();
//       return newTags;
//     } catch { return null; }
//   };
//   const removeTagFromImage = async (imgId, tag) => {
//     const fd=new FormData(); fd.append("tag",tag);
//     try {
//       const r = await axios.post(`${API}/photo/${imgId}/tags/remove`,fd);
//       const newTags = r.data.tags || [];
//       setImages(prev => prev.map(i => i.id===imgId ? {...i, user_tags: newTags} : i));
//       loadTags();
//       return newTags;
//     } catch { return null; }
//   };

//   const isGridView = ["timeline","search","favorites","explore"].includes(view);

//   const NAV = [
//     { id:"timeline",   icon:<Clock size={15}/>,     label:"Timeline"   },
//     { id:"albums",     icon:<BookOpen size={15}/>,  label:"Albums"     },
//     { id:"faces",      icon:<Users size={15}/>,     label:"People"     },
//     { id:"favorites",  icon:<Heart size={15}/>,     label:"Favorites"  },
//     { id:"duplicates", icon:<Copy size={15}/>,      label:"Duplicates" },
//     { id:"explore",    icon:<Shuffle size={15}/>,   label:"Explore"    },
//     { id:"stats",      icon:<BarChart3 size={15}/>, label:"Statistics" },
//   ];
//   const MODES = [
//     { id:"text",     Icon:Search,   label:"Search"   },
//     { id:"describe", Icon:FileText, label:"Describe" },
//     { id:"image",    Icon:ImgIcon,  label:"By Image" },
//     { id:"hybrid",   Icon:GitMerge, label:"Hybrid"   },
//     { id:"color",    Icon:Palette,  label:"By Color" },
//   ];
//   const COLORS = ["red","orange","yellow","green","blue","purple","pink","white","black","gray","brown"];

//   return (
//     <>
//       <CSS />
//       <div className="shell">

//         {/* ── SIDEBAR ─────────────────────────────────────────────────── */}
//         <nav className="sidebar">
//           <div className="brand">
//             <div className="brand-mark"><Aperture size={16} strokeWidth={1.5}/></div>
//             <div>
//               <p className="brand-name">LUMINA</p>
//               <p className="brand-sub">AI Gallery</p>
//             </div>
//           </div>

//           <div className="nav-list">
//             {NAV.map(n=>(
//               <button key={n.id} className={`nav-btn ${view===n.id?"nav-btn--on":""}`} onClick={()=>setView(n.id)}>
//                 <span className="nav-ic">{n.icon}</span>
//                 <span>{n.label}</span>
//                 {view===n.id && <span className="nav-pip"/>}
//               </button>
//             ))}
//           </div>

//           <div className="sidebar-sep"/>

//           {/* Advanced Filters */}
//           <p className="sidebar-section-label"><SlidersHorizontal size={10}/> FILTERS</p>
//           <button className="nav-btn" style={{fontSize:11}} onClick={()=>filterGroupPhotos(2)}>
//             <span className="nav-ic"><Users size={14}/></span><span>2+ People</span>
//           </button>
//           <button className="nav-btn" style={{fontSize:11}} onClick={()=>filterGroupPhotos(3)}>
//             <span className="nav-ic"><Users size={14}/></span><span>3+ People</span>
//           </button>
//           <button className="nav-btn" style={{fontSize:11}} onClick={()=>filterGroupPhotos(5)}>
//             <span className="nav-ic"><Users size={14}/></span><span>5+ People</span>
//           </button>
//           <button className="nav-btn" style={{fontSize:11}} onClick={()=>setShowCoModal(true)}>
//             <span className="nav-ic"><GitMerge size={14}/></span><span>Co-Appear</span>
//           </button>
//           <button className="nav-btn" style={{fontSize:11}} onClick={()=>setShowFaceSim(true)}>
//             <span className="nav-ic"><UserCheck size={14}/></span><span>Face Search</span>
//           </button>

//           <div className="sidebar-sep"/>

//           <button className={`nav-btn nav-btn--trash ${view==="trash"?"nav-btn--on":""}`} onClick={()=>setView("trash")}>
//             <span className="nav-ic"><Trash size={15}/></span>
//             <span>Trash</span>
//             {trash.length>0 && <span className="badge">{trash.length}</span>}
//           </button>

//           {/* Tags sidebar section */}
//           {allTags.length>0 && (
//             <>
//               <div className="sidebar-sep"/>
//               <p className="sidebar-section-label"><Hash size={10}/> TAGS</p>
//               <div className="tag-nav-list">
//                 {allTags.slice(0,8).map(t=>(
//                   <button key={t.tag} className={`tag-nav-btn ${activeTag===t.tag?"tag-nav-btn--on":""}`} onClick={()=>filterByTag(t.tag)}>
//                     <span>#{t.tag}</span><span className="tag-nav-count">{t.count}</span>
//                   </button>
//                 ))}
//               </div>
//             </>
//           )}

//           <div className="sidebar-sep"/>
//           <div className="sidebar-foot">
//             <button className="reindex-btn reindex-btn--primary" onClick={doReindex} disabled={reindexing}>
//               <Zap size={13} className={reindexing?"spin":""}/>{reindexing?"Indexing…":"Re-index AI"}
//             </button>
//             <ToolsAccordion reindexing={reindexing}
//               onEmotions={reprocessEmotions}
//               onColors={reprocessColors}
//               onNames={reprocessNames}
//               onCaptions={()=>reprocessCaptions(false)}
//               onRecaptionAll={()=>reprocessCaptions(true)}
//               onCleanup={async()=>{ if(!window.confirm("Delete all empty albums?")) return; const r=await axios.delete(`${API}/albums/empty/cleanup`); setReindexMsg(`✓ Deleted ${r.data.deleted} empty albums`); load(); }}
//             />
//             {reindexMsg && <p className="reindex-msg">{reindexMsg}</p>}
//             <div className="offline-pill"><span className="dot-green"/><span>Fully Offline</span></div>
//             {voiceErr && voiceErr.includes("setup") && (
//               <div style={{background:"rgba(245,158,11,.08)",border:"1px solid rgba(245,158,11,.2)",borderRadius:7,padding:"8px 10px",fontSize:10,color:"rgba(245,158,11,.9)",lineHeight:1.5}}>
//                 <strong>Voice Setup:</strong><br/>
//                 1. pip install vosk<br/>
//                 2. Download model from<br/>
//                 <span style={{color:"#60a5fa",wordBreak:"break-all"}}>alphacephei.com/vosk/models</span><br/>
//                 3. Extract to:<br/>
//                 <code style={{fontSize:9,color:"#aaa"}}>models/vosk-model-small-en-us</code>
//               </div>
//             )}
//           </div>
//         </nav>

//         {/* ── MAIN ────────────────────────────────────────────────────── */}
//         <div className="main">

//           {/* TOPBAR */}
//           <header className="topbar">
//             <div className="mode-strip">
//               {MODES.map(({id,Icon,label})=>(
//                 <button key={id} className={`mode-btn ${searchMode===id?"mode-btn--on":""}`} onClick={()=>setSearchMode(id)}>
//                   <Icon size={11}/> {label}
//                 </button>
//               ))}
//               {voice.state!=="idle" && (
//                 <span className={`voice-chip voice-chip--${voice.state}`}>
//                   {voice.state==="rec"        && <><MicOff size={10}/> Recording</>}
//                   {voice.state==="processing" && <><Loader2 size={10} className="spin"/> Transcribing…</>}
//                   {voice.state==="error"      && <span title={voiceErr}>{voiceErr.length>38?voiceErr.slice(0,38)+"…":voiceErr}</span>}
//                 </span>
//               )}
//             </div>
//             {/* ── Quick emoji search chips ───────────────────────────────── */}
//             <div className="emoji-quick-bar">
//               <span className="emoji-quick-label">Quick:</span>
//               {[["😊","happy"],["😢","sad"],["😠","angry"],["😲","surprised"],["😨","fearful"],
//                 ["🐶","dog"],["🏖️","beach"],["🌅","sunset"],["🎉","party"],["✈️","travel"],
//                 ["❄️","snow"],["🌸","flowers"],["🍕","food"],["🏋️","gym"],["🌊","ocean"]
//               ].map(([emoji,label])=>(
//                 <button key={emoji} className="emoji-quick-btn" title={label}
//                   onClick={()=>{ setSearchQ(emoji); setSearchMode("text"); runSearch("text", emoji); }}>
//                   {emoji}
//                 </button>
//               ))}
//             </div>
//             <div className="search-bar">
//               {(searchMode==="text"||searchMode==="describe"||searchMode==="hybrid") && (
//                 <div className="search-wrap">
//                   <Search size={14} className="search-ic"/>
//                   <input className="search-in" value={searchQ} onChange={e=>setSearchQ(e.target.value)}
//                     onKeyDown={e=>e.key==="Enter"&&runSearch()}
//                     placeholder={searchMode==="describe"?"Describe what's in the photo…":searchMode==="hybrid"?"Text + optional image…":"Search by scene, object, person, text…"}
//                   />
//                   {voice.ok && (
//                     <button className={`mic-btn ${voice.state==="rec"?"mic-btn--rec":voice.state==="processing"?"mic-btn--proc":""}`}
//                       onClick={voice.state==="rec"?voice.stop:voice.start}>
//                       {voice.state==="rec"?<MicOff size={14}/>:voice.state==="processing"?<Loader2 size={14} className="spin"/>:<Mic size={14}/>}
//                     </button>
//                   )}
//                 </div>
//               )}
//               {(searchMode==="image"||searchMode==="hybrid") && (
//                 <label className="img-search-label">
//                   {hybridPrev?<img src={hybridPrev} className="img-search-thumb" alt=""/>:<ImgIcon size={13}/>}
//                   <span>{hybridPrev?"Ready":"Upload ref"}</span>
//                   <input type="file" accept="image/*" style={{display:"none"}} onChange={e=>{const f=e.target.files?.[0];if(f){setHybridFile(f);setHybridPrev(URL.createObjectURL(f))}}}/>
//                 </label>
//               )}
//               {searchMode==="color" && (
//                 <div className="color-strip">
//                   {COLORS.map(c=>(
//                     <button key={c} title={c} className={`clr-dot ${colorPick===c?"clr-dot--on":""}`}
//                       style={{background:c==="white"?"#e5e5e5":c==="gray"?"#888":c}}
//                       onClick={()=>setColorPick(c)}/>
//                   ))}
//                 </div>
//               )}
//               <button className="btn-go" onClick={()=>runSearch()}>Search</button>
//               {isGridView && (
//                 <button className={`btn-batch ${batchMode?"btn-batch--on":""}`} onClick={()=>{setBatchMode(p=>!p);clearSel();}}>
//                   <CheckSquare size={13}/> {batchMode?"Exit Select":"Select"}
//                 </button>
//               )}
// <button className="btn-upload" onClick={()=>uploadRef.current?.click()}>
//                 <Upload size={13}/> Upload
//               </button>
//               <input ref={uploadRef} type="file" multiple accept="image/*" style={{display:"none"}} onChange={e=>handleUpload(e.target.files)}/>
//             </div>

//             {/* Emotion filter bar */}
//             {isGridView && emoSummary.length>0 && (
//               <div className="emo-filter-bar">
//                 <span className="emo-filter-label">Filter:</span>
//                 {emoSummary.filter(e=>e.emotion&&e.emotion!=="neutral").map(e=>(
//                   <button key={e.emotion}
//                     className={`emo-filter-btn ${emoFilter===e.emotion?"emo-filter-btn--on":""}`}
//                     style={emoFilter===e.emotion?{background:EMO_COLORS[e.emotion]+"22",borderColor:EMO_COLORS[e.emotion]+"66",color:EMO_COLORS[e.emotion]}:{}}
//                     onClick={()=>filterByEmotion(e.emotion)}>
//                     {EMO[e.emotion]||"😐"} {e.emotion} <span className="emo-filter-count">{e.count}</span>
//                   </button>
//                 ))}
//                 {(emoFilter||activeTag) && (
//                   <button className="emo-filter-clear" onClick={()=>{setEmoFilter("");setActiveTag("");load();}}>
//                     <X size={10}/> Clear
//                   </button>
//                 )}
//               </div>
//             )}
//           </header>

//           {/* BODY */}
//           <div className="body">
//             <AnimatePresence mode="wait">
//               {loading ? (
//                 <Fade key="load">
//                   <div className="loader-wrap"><div className="loader-ring"/><p className="loader-text">Loading…</p></div>
//                 </Fade>

//               ) : isGridView ? (
//                 <Fade key={view+emoFilter+activeTag}>
//                   {/* On This Day banner */}
//                   {view==="timeline" && onThisDay?.total>0 && (
//                     <OnThisDayBanner data={onThisDay} expanded={otdExpanded} onToggle={()=>setOtdExpanded(p=>!p)} onOpen={openLb}/>
//                   )}

//                   <div className="page-head">
//                     <h1 className="page-title">
//                       {view==="search"?`Results for "${searchQ}"`:view==="favorites"?"Favorites":view==="explore"?"Explore":
//                        activeTag?`#${activeTag}`:emoFilter?`${EMO[emoFilter]} ${emoFilter}`:"Timeline"}
//                     </h1>
//                     {images.length>0 && <span className="page-count">{images.length} photos</span>}
//                     {batchMode && selected.size>0 && <span className="sel-count">{selected.size} selected</span>}
//                     {view==="explore" && <button className="btn-sm" style={{marginLeft:"auto"}} onClick={load}><Shuffle size={11}/> Shuffle</button>}
//                   </div>

//                   {images.length===0
//                     ? (view==="search" && peopleSuggestions.length>0 ? (
//                         <div>
//                           <Empty icon={ImageOff} msg="No photos matched" sub="But these people are in your gallery — click to view their photos"/>
//                           <div style={{marginTop:16}}>
//                             <p style={{fontSize:11,color:"var(--mu)",marginBottom:10,fontFamily:"var(--mono)",letterSpacing:".08em"}}>PEOPLE IN YOUR GALLERY:</p>
//                             <div className="people-grid">
//                               {peopleSuggestions.map(p=>(
//                                 <motion.div key={p.id} className="person-card" whileHover={{y:-3}}
//                                   onClick={async()=>{
//                                     setLoading(true); setView("search"); setPeopleSuggestions([]);
//                                     try { const r=await axios.get(`${API}/people/${p.id}`); setImages(r.data.results||r.data.images||[]); }
//                                     catch {} finally { setLoading(false); }
//                                   }}>
//                                   <div className="person-avatar">
//                                     {p.cover?<img src={imgUrl(p.cover)} alt="" onError={e=>e.target.src=BLANK}/>:<Users size={26} strokeWidth={1} color="#444"/>}
//                                   </div>
//                                   <p className="person-name">{p.name}</p>
//                                   <p className="person-count" style={{color:"var(--ac2)",fontSize:9}}>click to rename</p>
//                                 </motion.div>
//                               ))}
//                             </div>
//                             <p style={{fontSize:11,color:"#555",marginTop:14,padding:"10px 14px",background:"var(--s2)",borderRadius:8,border:"1px solid var(--br)"}}>
//                               💡 <strong>Tip:</strong> Click a person above → click ✏ Rename → type their real name → search will work instantly
//                             </p>
//                           </div>
//                         </div>
//                       ) : <Empty icon={ImageOff} msg={view==="search"?"No results":"No photos yet"} sub={view!=="search"?"Upload photos to get started":"Try different keywords"} onClick={view!=="search"?()=>uploadRef.current?.click():null}/>)
//                     : <ImgGrid list={images} onOpen={i=>openLb(images,i)} onFav={toggleFav} onDel={softDelete} batchMode={batchMode} selected={selected} onToggle={toggleSelect}/>
//                   }
//                 </Fade>

//               ) : view==="faces" ? (
//                 <Fade key="faces">
//                   <PageHead title="People" count={faces.length}/>
//                   {faces.length===0
//                     ? <Empty icon={Users} msg="No people detected" sub="Run Re-index to detect faces"/>
//                     : <div className="people-grid">{faces.map(p=><PersonCard key={p.id} p={p} onClick={()=>openPerson(p)}/>)}</div>
//                   }
//                 </Fade>

//               ) : view==="albums" ? (
//                 <Fade key="albums">
//                   <div className="page-head">
//                     <h1 className="page-title">Albums</h1>
//                     {albums.length>0 && <span className="page-count">{albums.length}</span>}
//                     <button className="btn-sm btn-sm--accent" style={{marginLeft:"auto"}} onClick={()=>{ setPendingAlbumImages([]); setCreateAlbumOpen(true); }}>
//                       <Plus size={12}/> New Album
//                     </button>
//                     <button className="btn-sm" style={{marginLeft:6,background:"var(--s3)",border:"1px solid var(--br)"}} onClick={()=>setCreateEventOpen(true)}>🎉 New Event</button>
//                   </div>
//                   {albums.length===0
//                     ? <Empty icon={BookImage} msg="No albums" sub="Run Re-index to auto-generate or create manually"/>
//                     : <div className="albums-grid">{albums.map(a=><AlbumCard key={a.id} a={a} onClick={()=>openAlbum(a)} onDelete={a.type==="manual"?()=>deleteAlbum(a.id):null}/>)}</div>
//                   }
//                 </Fade>

//               ) : view==="duplicates" ? (
//                 <Fade key="dupes">
//                   <PageHead title="Duplicate Groups" count={dupes.length} unit="groups"/>
//                   {dupes.length===0
//                     ? <Empty icon={Copy} msg="No duplicates" sub="All images are unique"/>
//                     : <div className="dupe-list">
//                         {dupes.map((g,gi)=>(
//                           <div key={gi} className="dupe-group">
//                             <div className="dupe-head">
//                               <span className="tag-sm">Group {gi+1}</span>
//                               <span className="tag-sm tag-sm--accent">{g.count} duplicates</span>
//                               {g.total_size&&<span className="tag-sm">{(g.total_size/1024/1024).toFixed(1)} MB</span>}
//                             </div>
//                             <ImgGrid list={g.images||[]} compact onOpen={i=>openLb(g.images,i)} onFav={toggleFav} onDel={softDelete}/>
//                           </div>
//                         ))}
//                       </div>
//                   }
//                 </Fade>

//               ) : view==="trash" ? (
//                 <Fade key="trash">
//                   <PageHead title="Trash" count={trash.length}/>
//                   {trash.length===0 ? <Empty icon={Trash} msg="Trash is empty"/> : (
//                     <div className="img-grid">
//                       {trash.map(img=>(
//                         <div key={img.id} className="trash-card">
//                           <img src={imgUrl(img.filename)} alt="" className="trash-img" onError={e=>e.target.src=BLANK}/>
//                           <div className="trash-overlay">
//                             <button className="trash-btn trash-btn--restore" onClick={()=>restoreImg(img.id)}><RotateCcw size={11}/> Restore</button>
//                             <button className="trash-btn trash-btn--del" onClick={()=>permDelete(img.id)}><AlertTriangle size={11}/> Delete Forever</button>
//                           </div>
//                           {img.trashed_at&&<span className="trash-date">{new Date(img.trashed_at).toLocaleDateString()}</span>}
//                         </div>
//                       ))}
//                     </div>
//                   )}
//                 </Fade>

//               ) : view==="stats" ? (
//                 <Fade key="stats"><StatsPage stats={stats} peopleFreq={peopleFreq}/></Fade>
//               ) : null}
//             </AnimatePresence>
//           </div>

//           {/* BATCH ACTION BAR */}
//           <AnimatePresence>
//             {batchMode && (
//               <motion.div className="batch-bar"
//                 initial={{y:60,opacity:0}} animate={{y:0,opacity:1}} exit={{y:60,opacity:0}}>
//                 <div className="batch-bar-left">
//                   <span className="batch-count"><CheckCheck size={14}/> {selected.size} selected</span>
//                   <button className="batch-sm" onClick={selectAll}>All</button>
//                   <button className="batch-sm" onClick={clearSel}>None</button>
//                 </div>
//                 <div className="batch-bar-actions">
//                   <button className="batch-action-btn batch-action-btn--fav" onClick={()=>batchFavorite(true)} title="Favorite all"><Heart size={13}/> Favorite</button>
//                   <button className="batch-action-btn batch-action-btn--del" onClick={batchDelete} title="Trash all"><Trash2 size={13}/> Trash</button>
//                   <div className="batch-input-group">
//                     <input className="batch-input" placeholder="Add tag…" value={batchTagInput} onChange={e=>setBatchTagInput(e.target.value)} onKeyDown={e=>e.key==="Enter"&&batchTag()}/>
//                     <button className="batch-action-btn" onClick={batchTag}><Tag size={12}/></button>
//                   </div>
//                   <div className="batch-input-group">
//                     <select className="batch-select" value={batchAlbumId} onChange={e=>setBatchAlbumId(e.target.value)}>
//                       <option value="">Add to album…</option>
//                       {albums.map(a=><option key={a.id} value={a.id}>{a.title}</option>)}
//                     </select>
//                     <button className="batch-action-btn" onClick={batchAddAlbum} title="Add to existing album"><Layers size={12}/></button>
//                   </div>
//                   <button className="batch-action-btn batch-action-btn--album"
//                     onClick={()=>{
//                       setPendingAlbumImages([...selected]); // snapshot NOW before modal opens
//                       setCreateAlbumOpen(true);
//                     }}
//                     title={`Create new album with ${selected.size} selected photos`}
//                     disabled={selected.size===0}>
//                     <FolderPlus size={12}/> New Album
//                   </button>
//                 </div>
//               </motion.div>
//             )}
//           </AnimatePresence>
//         </div>
//       </div>

//       {/* LIGHTBOX */}
//       <AnimatePresence>
//         {lb && <Lightbox list={lb.list} idx={lb.idx} onClose={closeLb} onShift={shiftLb} onFav={toggleFav} onDel={(id,e)=>{softDelete(id,e);closeLb();}} onAddTag={addTagToImage} onRemoveTag={removeTagFromImage} albums={albums}/>}
//       </AnimatePresence>

//       {/* PERSON MODAL */}
//       <AnimatePresence>
//         {personModal && <PersonModal data={personModal} onClose={()=>setPersonModal(null)} onRename={renamePerson} onOpen={(list,i)=>{setPersonModal(null);setTimeout(()=>openLb(list,i),80);}}/>}
//       </AnimatePresence>

//       {/* ALBUM MODAL */}
//       <AnimatePresence>
//         {albumModal && <AlbumModal data={albumModal} onClose={()=>setAlbumModal(null)} onRename={renameAlbum} onDelete={deleteAlbum} onOpen={(list,i)=>{setAlbumModal(null);setTimeout(()=>openLb(list,i),80);}}/>}
//       </AnimatePresence>

//       {/* CREATE ALBUM MODAL */}
//       <AnimatePresence>
//         {createAlbumOpen && (
//           <ModalWrap onClose={()=>setCreateAlbumOpen(false)}>
//             <div className="modal-head">
//               <h2 className="modal-title">New Album</h2>
//               <button className="lb-hbtn lb-close" onClick={()=>setCreateAlbumOpen(false)}><X size={17}/></button>
//             </div>
//             {pendingAlbumImages.length>0 && (
//               <div style={{background:"rgba(96,165,250,.08)",border:"1px solid rgba(96,165,250,.2)",borderRadius:8,padding:"8px 12px",marginBottom:12,fontSize:12,color:"rgba(96,165,250,.9)"}}>
//                 📸 <strong>{pendingAlbumImages.length} selected photos</strong> will be added to this album
//               </div>
//             )}
//             <div className="create-album-form">
//               <input className="form-input" placeholder="Album title…" value={newAlbumTitle}
//                 onChange={e=>setNewAlbumTitle(e.target.value)}
//                 onKeyDown={e=>e.key==="Enter"&&newAlbumTitle.trim()&&createAlbum()}
//                 autoFocus/>
//               <textarea className="form-textarea" placeholder="Description (optional)…"
//                 value={newAlbumDesc} onChange={e=>setNewAlbumDesc(e.target.value)} rows={2}/>
//               <div style={{display:"flex",gap:8}}>
//                 <button className="btn-sm btn-sm--primary" style={{flex:1}} onClick={()=>createAlbum()} disabled={!newAlbumTitle.trim()}>
//                   <FolderPlus size={13}/> Create Album
//                 </button>
//                 <button className="btn-sm" onClick={()=>setCreateAlbumOpen(false)}>Cancel</button>
//               </div>
//             </div>
//           </ModalWrap>
//         )}
//       </AnimatePresence>

//       {/* CREATE NAMED EVENT MODAL */}
//       <AnimatePresence>
//         {createEventOpen && (
//           <ModalWrap onClose={()=>setCreateEventOpen(false)}>
//             <div className="modal-head">
//               <h2 className="modal-title">🎉 New Event</h2>
//               <button className="lb-hbtn lb-close" onClick={()=>setCreateEventOpen(false)}><X size={17}/></button>
//             </div>
//             <div className="create-album-form">
//               <label style={{fontSize:11,color:"var(--mu)",marginBottom:4}}>EVENT TYPE</label>
//               <div style={{display:"flex",flexWrap:"wrap",gap:6,marginBottom:12}}>
//                 {["Birthday","Trip","Vacation","Wedding","Anniversary","Graduation","Party","Holiday","Family","Work","Other"].map(t=>(
//                   <button key={t}
//                     className={"btn-sm" + (newEventType===t?" btn-sm--primary":"")}
//                     style={{fontSize:11,padding:"4px 10px"}}
//                     onClick={()=>setNewEventType(t)}>
//                     {t==="Birthday"?"🎂":t==="Trip"?"✈️":t==="Vacation"?"🏖️":t==="Wedding"?"💍":t==="Anniversary"?"💕":t==="Graduation"?"🎓":t==="Party"?"🎊":t==="Holiday"?"🎄":t==="Family"?"👨‍👩‍👧":t==="Work"?"💼":"📅"} {t}
//                   </button>
//                 ))}
//               </div>
//               <input className="form-input" placeholder="Event name (e.g. Mom's 60th Birthday)…" value={newEventTitle} onChange={e=>setNewEventTitle(e.target.value)} autoFocus/>
//               <input className="form-input" type="date" value={newEventDate} onChange={e=>setNewEventDate(e.target.value)} style={{marginTop:8}}/>
//               <textarea className="form-textarea" placeholder="Notes (optional)…" value={newEventDesc} onChange={e=>setNewEventDesc(e.target.value)} rows={2} style={{marginTop:8}}/>
//               <button className="btn-sm btn-sm--primary" onClick={createNamedEvent} disabled={!newEventTitle.trim()} style={{marginTop:4}}>
//                 🎉 Create Event
//               </button>
//             </div>
//           </ModalWrap>
//         )}
//       </AnimatePresence>

//       {/* CO-OCCURRENCE MODAL */}
//       <AnimatePresence>
//         {showCoModal && (
//           <ModalWrap onClose={()=>setShowCoModal(false)}>
//             <div className="modal-head">
//               <h2 className="modal-title"><GitMerge size={16} style={{marginRight:6,verticalAlign:"middle"}}/>Co-Appearance Search</h2>
//               <button className="lb-hbtn lb-close" onClick={()=>setShowCoModal(false)}><X size={17}/></button>
//             </div>
//             <CoOccurrencePanel faces={faces} onSearch={runCoOccurrence}/>
//           </ModalWrap>
//         )}
//       </AnimatePresence>

//       {/* FACE SIMILARITY MODAL */}
//       <AnimatePresence>
//         {showFaceSim && (
//           <ModalWrap onClose={()=>setShowFaceSim(false)}>
//             <div className="modal-head">
//               <h2 className="modal-title"><UserCheck size={16} style={{marginRight:6,verticalAlign:"middle"}}/>Face Similarity Search</h2>
//               <button className="lb-hbtn lb-close" onClick={()=>setShowFaceSim(false)}><X size={17}/></button>
//             </div>
//             <div className="create-album-form">
//               <p style={{fontSize:12,color:"var(--mu)",marginBottom:12}}>Upload a face crop or photo — find all photos with the same person.</p>
//               {faceSrcPrev && <img src={faceSrcPrev} style={{width:120,height:120,objectFit:"cover",borderRadius:8,marginBottom:12,border:"2px solid var(--ac)"}} alt="face"/>}
//               <input type="file" accept="image/*" style={{display:"none"}} id="face-sim-input"
//                 onChange={e=>{
//                   const f=e.target.files[0]; if(!f) return;
//                   setFaceSrcFile(f);
//                   const r=new FileReader(); r.onload=ev=>setFaceSrcPrev(ev.target.result); r.readAsDataURL(f);
//                 }}/>
//               <button className="btn-sm" onClick={()=>document.getElementById("face-sim-input").click()}>
//                 <Camera size={12}/> {faceSrcPrev?"Change Photo":"Upload Face Photo"}
//               </button>
//               {faceSrcFile && (
//                 <button className="btn-sm btn-sm--primary" style={{marginTop:10}} onClick={runFaceSimilarity}>
//                   <UserCheck size={12}/> Find Similar Faces
//                 </button>
//               )}
//             </div>
//           </ModalWrap>
//         )}
//       </AnimatePresence>
//     </>
//   );
// }

// /* ══════════════════════════════════════════════════════════════════════════ */
// /* ON THIS DAY BANNER                                                          */
// /* ══════════════════════════════════════════════════════════════════════════ */
// function OnThisDayBanner({ data, expanded, onToggle, onOpen }) {
//   return (
//     <div className="otd-banner">
//       <div className="otd-header" onClick={onToggle}>
//         <div className="otd-left">
//           <Calendar size={15} color="#f59e0b"/>
//           <div>
//             <span className="otd-title">On This Day — {data.date}</span>
//             <span className="otd-sub">{data.total} photo{data.total!==1?"s":""} from {data.years?.length} year{data.years?.length!==1?"s":""} ago</span>
//           </div>
//         </div>
//         <button className="otd-toggle">{expanded?<ChevronUp size={14}/>:<ChevronDown size={14}/>}</button>
//       </div>
//       <AnimatePresence>
//         {expanded && (
//           <motion.div initial={{height:0,opacity:0}} animate={{height:"auto",opacity:1}} exit={{height:0,opacity:0}}>
//             {data.years?.map(yr=>(
//               <div key={yr.year} className="otd-year">
//                 <p className="otd-year-label">{yr.year} — {yr.count} photo{yr.count!==1?"s":""}</p>
//                 <div className="otd-strip">
//                   {yr.images.map((img,i)=>(
//                     <div key={img.id} className="otd-thumb" onClick={()=>onOpen(yr.images,i)}>
//                       <img src={imgUrl(img.filename)} alt="" onError={e=>e.target.src=BLANK}/>
//                       {img.caption_short && <div className="otd-thumb-caption">{img.caption_short}</div>}
//                     </div>
//                   ))}
//                 </div>
//               </div>
//             ))}
//           </motion.div>
//         )}
//       </AnimatePresence>
//     </div>
//   );
// }


// /* ══════════════════════════════════════════════════════════════════════════ */
// /* IMAGE GRID                                                                  */
// /* ══════════════════════════════════════════════════════════════════════════ */
// function ImgGrid({ list, onOpen, onFav, onDel, compact, batchMode, selected, onToggle }) {
//   return (
//     <div className={`img-grid${compact?" img-grid--sm":""}`}>
//       {list.map((img,i)=>(
//         <PhotoCard key={img.id??i} img={img} onOpen={()=>onOpen(i)} onFav={onFav} onDel={onDel}
//           batchMode={batchMode} isSelected={selected?.has(img.id)} onToggle={onToggle}/>
//       ))}
//     </div>
//   );
// }

// function PhotoCard({ img, onOpen, onFav, onDel, batchMode, isSelected, onToggle }) {
//   const q=qInfo(img.quality_level); const emo=EMO[img.dominant_emotion]||"";
//   return (
//     <motion.div
//       className={`photo-card ${isSelected?"photo-card--sel":""}`}
//       whileHover={{y:-4, scale:1.02}}
//       whileTap={{scale:0.97}}
//       transition={{type:"spring", stiffness:380, damping:28}}
//       onClick={batchMode?()=>onToggle(img.id):onOpen}
//     >
//       <img src={imgUrl(img.filename)||BLANK} alt="" className="photo-img" loading="lazy"
//         onError={e=>{e.target.onerror=null;e.target.src=BLANK;}}/>

//       {/* Top chips */}
//       {img.quality_level&&img.quality_level!=="Processing..."&&(
//         <span className="chip chip--q" style={{color:q.c,background:q.bg,borderColor:q.c+"44"}}>{img.quality_level[0]}</span>
//       )}
//       {emo&&img.dominant_emotion!=="neutral"&&<span className="chip chip--emo">{emo}</span>}
//       {img.is_favorite&&<span className="fav-dot">♥</span>}

//       {batchMode ? (
//         <motion.div className="batch-check" initial={{scale:0.5}} animate={{scale:1}}>
//           {isSelected
//             ? <CheckSquare size={20} color="#7c6af7" style={{filter:"drop-shadow(0 0 6px #7c6af7aa)"}}/>
//             : <Square size={20} color="rgba(255,255,255,.4)"/>
//           }
//         </motion.div>
//       ) : (
//         <div className="photo-hover">
//           <div className="photo-actions">
//             <motion.button whileHover={{scale:1.15}} whileTap={{scale:.9}}
//               className="ph-btn ph-btn--fav" onClick={e=>{e.stopPropagation();onFav(img.id,e);}}>
//               <Heart size={12} fill={img.is_favorite?"#fff":"none"}/>
//             </motion.button>
//             <motion.button whileHover={{scale:1.15}} whileTap={{scale:.9}}
//               className="ph-btn ph-btn--del" onClick={e=>{e.stopPropagation();onDel(img.id,e);}}>
//               <Trash2 size={12}/>
//             </motion.button>
//           </div>
//           <div className="photo-info">
//             {img.caption_short&&<p className="photo-caption">{img.caption_short}</p>}
//             <div className="photo-meta-row">
//               {img.quality_level&&img.quality_level!=="Processing..."&&
//                 <span className="meta-chip" style={{color:q.c}}>◆ {img.quality_level}</span>}
//               {img.aesthetic_score>0&&
//                 <span className="meta-chip" style={{color:"#fbbf24"}}>★ {Number(img.aesthetic_score).toFixed(1)}</span>}
//               {img.dominant_emotion&&img.dominant_emotion!=="neutral"&&
//                 <span className="meta-chip">{emo} {img.dominant_emotion}</span>}
//               {img.user_tags?.length>0&&
//                 <span className="meta-chip" style={{color:"#86efac"}}>#{img.user_tags[0]}{img.user_tags.length>1?` +${img.user_tags.length-1}`:""}</span>}
//             </div>
//           </div>
//         </div>
//       )}
//     </motion.div>
//   );
// }

// /* ══════════════════════════════════════════════════════════════════════════ */
// /* LIGHTBOX                                                                    */
// /* ══════════════════════════════════════════════════════════════════════════ */
// function Lightbox({ list, idx, onClose, onShift, onFav, onDel, onAddTag, onRemoveTag, albums }) {
//   const img=list[idx]; if(!img) return null;
//   const q=qInfo(img.quality_level); const emo=EMO[img.dominant_emotion]||"😐";
//   const [tagInput, setTagInput] = useState("");
//   const [imgTags,  setImgTags]  = useState(img.user_tags||[]);
//   const [addAlbum, setAddAlbum] = useState("");
//   const [note,     setNote]     = useState(img.photo_note||"");
//   const [noteEditing, setNoteEditing] = useState(false);
//   const [noteSaving,  setNoteSaving]  = useState(false);

//   // Always fetch latest tags + note from server when photo changes
//   useEffect(()=>{
//     setTagInput(""); setNoteEditing(false);
//     setImgTags(img.user_tags||[]);
//     setNote(img.photo_note||"");
//     axios.get(`${API}/photo/${img.id}/tags`)
//       .then(r => setImgTags(r.data.tags||[]))
//       .catch(()=>{});
//     axios.get(`${API}/photo/${img.id}/note`)
//       .then(r => setNote(r.data.note||""))
//       .catch(()=>{});
//   }, [img.id]);

//   const saveNote = async () => {
//     setNoteSaving(true);
//     const fd=new FormData(); fd.append("note", note);
//     try { await axios.post(`${API}/photo/${img.id}/note`, fd); setNoteEditing(false); }
//     catch(e) { alert("Failed to save note"); }
//     setNoteSaving(false);
//   };

//   const doAddTag = async () => {
//     if (!tagInput.trim()) return;
//     const newTags = await onAddTag(img.id, tagInput.trim());
//     if (newTags) setImgTags(newTags);
//     else setImgTags(p=>[...new Set([...p, tagInput.trim().toLowerCase()])]);
//     setTagInput("");
//   };
//   const doRemoveTag = async tag => {
//     const newTags = await onRemoveTag(img.id, tag);
//     if (newTags) setImgTags(newTags);
//     else setImgTags(p=>p.filter(t=>t!==tag));
//   };
//   const doAddToAlbum = async () => {
//     if (!addAlbum) return;
//     const fd=new FormData(); fd.append("image_ids",img.id);
//     try { await axios.post(`${API}/batch/album`,fd.set?fd:(()=>{fd.append("album_id",addAlbum);return fd;})()); } catch {}
//     setAddAlbum("");
//   };

//   return (
//     <motion.div className="lb-bg" initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}} onClick={onClose}>
//       <motion.div className="lb-box" initial={{scale:.96}} animate={{scale:1}} exit={{scale:.96}} onClick={e=>e.stopPropagation()}>
//         <div className="lb-left">
//           {idx>0&&<button className="lb-nav lb-nav--l" onClick={()=>onShift(-1)}><ChevronLeft size={20}/></button>}
//           <AnimatePresence mode="wait">
//             <motion.img key={img.id??idx} src={imgUrl(img.filename)||BLANK} alt="" className="lb-img"
//               initial={{opacity:0,x:16}} animate={{opacity:1,x:0}} exit={{opacity:0,x:-16}} transition={{duration:.18}}
//               onError={e=>{e.target.onerror=null;e.target.src=BLANK;}}/>
//           </AnimatePresence>
//           {idx<list.length-1&&<button className="lb-nav lb-nav--r" onClick={()=>onShift(1)}><ChevronRight size={20}/></button>}
//           <div className="lb-counter">{idx+1} / {list.length}</div>
//           <div className="lb-hint">← → · ESC</div>
//         </div>

//         <div className="lb-right">
//           <div className="lb-right-head">
//             <h3 className="lb-right-title">Details</h3>
//             <div className="lb-head-btns">
//               <button className="lb-hbtn" onClick={()=>onFav(img.id)} style={{color:img.is_favorite?"#f43f5e":undefined}}>
//                 <Heart size={15} fill={img.is_favorite?"#f43f5e":"none"} stroke={img.is_favorite?"#f43f5e":"currentColor"}/>
//               </button>
//               <button className="lb-hbtn" onClick={e=>onDel(img.id,e)}><Trash2 size={15}/></button>
//               <button className="lb-hbtn lb-close" onClick={onClose}><X size={17}/></button>
//             </div>
//           </div>

//           <div className="lb-scroll">
//             {img.caption_short&&(
//               <DB icon={<Camera size={12}/>} label="AI Caption" c="#818cf8">
//                 <p className="db-caption">{img.caption_short}</p>
//                 {img.caption_detailed&&img.caption_detailed!==img.caption_short&&<p className="db-caption-sub">{img.caption_detailed}</p>}
//               </DB>
//             )}
//             {img.quality_score>0&&(
//               <DB icon={<TrendingUp size={12}/>} label="Quality" c={q.c}>
//                 <div className="q-wrap">
//                   <div className="q-circle" style={{borderColor:q.c}}>
//                     <span style={{color:q.c,fontSize:18,fontWeight:800}}>{Math.round(img.quality_score)}</span>
//                   </div>
//                   <div className="q-right">
//                     <p className="q-level" style={{color:q.c}}>{img.quality_level}</p>
//                     {img.sharpness>0&&<><QBar label="Sharpness" val={img.sharpness} c={q.c}/><QBar label="Exposure" val={img.exposure} c={q.c}/><QBar label="Contrast" val={img.contrast} c={q.c}/><QBar label="Composition" val={img.composition} c={q.c}/></>}
//                   </div>
//                 </div>
//               </DB>
//             )}
//             {img.aesthetic_score>0&&(
//               <DB icon={<Star size={12}/>} label="Aesthetic" c="#fbbf24">
//                 <div className="aes-row"><span className="aes-score">★ {Number(img.aesthetic_score).toFixed(1)}</span>{img.aesthetic_rating&&<span className="aes-rating">{img.aesthetic_rating}</span>}</div>
//               </DB>
//             )}
//             {img.dominant_emotion&&img.dominant_emotion!=="neutral"&&(
//               <DB icon={<Smile size={12}/>} label="Emotion" c={EMO_COLORS[img.dominant_emotion]||"#f9a8d4"}>
//                 <div className="emo-row">
//                   <span style={{fontSize:30}}>{EMO[img.dominant_emotion]}</span>
//                   <div><p className="emo-name">{img.dominant_emotion}</p>{img.face_emotion_count>0&&<p className="emo-sub">{img.face_emotion_count} face{img.face_emotion_count>1?"s":""}</p>}</div>
//                 </div>
//               </DB>
//             )}
//             {img.ocr_text_enhanced?.trim()&&(
//               <DB icon={<Type size={12}/>} label="Text in Photo" c="#22d3ee">
//                 <p className="ocr-text">{img.ocr_text_enhanced}</p>
//               </DB>
//             )}

//             {/* Tags block */}
//             <DB icon={<FileText size={12}/>} label="Personal Note" c="#fcd34d">
//               {noteEditing ? (
//                 <div>
//                   <textarea className="form-textarea" value={note} onChange={e=>setNote(e.target.value)}
//                     placeholder="Write a personal note about this photo…" rows={3} autoFocus
//                     style={{marginBottom:6,fontSize:12}}/>
//                   <div style={{display:"flex",gap:6}}>
//                     <button className="btn-sm btn-sm--primary" onClick={saveNote} disabled={noteSaving}>
//                       {noteSaving?"Saving…":"Save Note"}
//                     </button>
//                     <button className="btn-sm" onClick={()=>setNoteEditing(false)}>Cancel</button>
//                   </div>
//                 </div>
//               ) : (
//                 <div>
//                   {note ? <p style={{fontSize:12,color:"var(--tx)",lineHeight:1.5,marginBottom:6}}>{note}</p>
//                         : <p style={{fontSize:11,color:"var(--mu)"}}>No note yet</p>}
//                   <button className="btn-sm" style={{fontSize:10,padding:"3px 8px"}} onClick={()=>setNoteEditing(true)}>
//                     ✏ {note?"Edit Note":"Add Note"}
//                   </button>
//                 </div>
//               )}
//             </DB>

//             <DB icon={<Hash size={12}/>} label="Tags" c="#86efac">
//               <div className="tag-chips">
//                 {imgTags.map(t=>(
//                   <span key={t} className="tag-chip-item">
//                     #{t}
//                     <button className="tag-chip-x" onClick={()=>doRemoveTag(t)}><X size={8}/></button>
//                   </span>
//                 ))}
//                 {imgTags.length===0&&<span style={{fontSize:10,color:"var(--mu)"}}>No tags yet</span>}
//               </div>
//               <div className="tag-add-row">
//                 <input className="tag-add-input" placeholder="Add tag…" value={tagInput} onChange={e=>setTagInput(e.target.value)} onKeyDown={e=>e.key==="Enter"&&doAddTag()}/>
//                 <button className="tag-add-btn" onClick={doAddTag}><Plus size={12}/></button>
//               </div>
//             </DB>

//             {/* Add to album */}
//             {albums?.length>0&&(
//               <DB icon={<Layers size={12}/>} label="Add to Album" c="#a78bfa">
//                 <div className="tag-add-row">
//                   <select className="batch-select" style={{flex:1}} value={addAlbum} onChange={e=>setAddAlbum(e.target.value)}>
//                     <option value="">Choose album…</option>
//                     {albums.map(a=><option key={a.id} value={a.id}>{a.title}</option>)}
//                   </select>
//                   <button className="tag-add-btn" onClick={doAddToAlbum}><Plus size={12}/></button>
//                 </div>
//               </DB>
//             )}

//             {img.scene_label&&(
//               <DB icon={<Tag size={12}/>} label="Objects" c="#86efac">
//                 <div className="tag-wrap">{img.scene_label.split(",").map((t,i)=><span key={i} className="obj-tag">{t.trim()}</span>)}</div>
//               </DB>
//             )}
//             <DB icon={<Info size={12}/>} label="Metadata" c="#6b7280">
//               <div className="meta-stack">
//                 {img.timestamp&&<MRow k="Captured" v={new Date(img.timestamp).toLocaleString()}/>}
//                 {img.width&&<MRow k="Dimensions" v={`${img.width} × ${img.height}`}/>}
//                 {img.person_count>0&&<MRow k="People" v={`${img.person_count} detected`}/>}
//                 {img.score>0&&<MRow k="Relevance" v={`${Math.round(img.score)}%`}/>}
//               </div>
//             </DB>
//           </div>
//         </div>
//       </motion.div>
//     </motion.div>
//   );
// }

// function QBar({label,val,c}){ return <div className="qbar-row"><span className="qbar-label">{label}</span><div className="qbar-track"><div className="qbar-fill" style={{width:`${Math.min(100,(val||0)*100)}%`,background:c}}/></div><span className="qbar-num">{Math.round((val||0)*100)}</span></div>; }
// function MRow({k,v}){ return <div className="mrow"><span className="mrow-k">{k}</span><span className="mrow-v">{v}</span></div>; }
// function DB({icon,label,c,children}){ return <div className="db" style={{"--dc":c}}><div className="db-head">{React.cloneElement(icon,{color:c})}<span className="db-label">{label}</span></div><div className="db-body">{children}</div></div>; }

// /* Person / Album components */
// function PersonCard({p,onClick}){ return <motion.div className="person-card" whileHover={{y:-3}} onClick={onClick}><div className="person-avatar">{p.cover?<img src={imgUrl(p.cover)} alt="" onError={e=>e.target.src=BLANK}/>:<Users size={26} strokeWidth={1} color="#444"/>}</div><p className="person-name">{p.name}</p><p className="person-count">{p.count||p.face_count||0} photos</p></motion.div>; }

// function PersonModal({data,onClose,onRename,onOpen}){
//   const [editing, setEditing] = useState(false);
//   const [name, setName] = useState(data.name || "");
//   const imgs = data.images || [];
//   const isDefault = /^Person \d+$/.test(data.name || "");

//   const doSave = () => {
//     if (name.trim()) { onRename(data.id, name.trim()); setEditing(false); }
//   };

//   return (
//     <ModalWrap onClose={onClose} wide>
//       <div className="modal-head">
//         <div>
//           <h2 className="modal-title">{data.name}</h2>
//           <p className="modal-sub">{data.face_count||data.count||0} photos</p>
//         </div>
//         <div style={{display:"flex",gap:6,alignItems:"center"}}>
//           <button className="btn-sm btn-sm--primary" onClick={()=>{setEditing(true);setName(isDefault?"":data.name);}}>
//             ✏ Rename
//           </button>
//           <button className="lb-hbtn lb-close" onClick={onClose}><X size={17}/></button>
//         </div>
//       </div>

//       {isDefault && !editing && (
//         <div style={{background:"rgba(250,204,21,.07)",border:"1px solid rgba(250,204,21,.2)",borderRadius:8,padding:"10px 14px",marginBottom:14}}>
//           <p style={{fontSize:12,color:"rgba(250,204,21,.9)"}}>
//             💡 <strong>Give this person a name</strong> so you can search for their photos. Click <strong>✏ Rename</strong> above.
//           </p>
//         </div>
//       )}

//       {editing && (
//         <div style={{background:"var(--s2)",border:"1px solid var(--br)",borderRadius:10,padding:"14px 16px",marginBottom:14}}>
//           <p style={{fontSize:11,color:"var(--mu)",marginBottom:8,fontFamily:"var(--mono)",letterSpacing:".07em"}}>
//             ENTER PERSON'S NAME
//           </p>
//           <div style={{display:"flex",gap:8,alignItems:"center"}}>
//             <input
//               className="rename-in"
//               style={{flex:1,fontSize:15,padding:"10px 14px"}}
//               value={name}
//               onChange={e=>setName(e.target.value)}
//               onKeyDown={e=>{if(e.key==="Enter")doSave(); if(e.key==="Escape")setEditing(false);}}
//               placeholder="e.g. Vijay, Shah Rukh Khan…"
//               autoFocus
//             />
//             <button className="btn-sm btn-sm--primary" style={{padding:"10px 18px",fontSize:13}} onClick={doSave} disabled={!name.trim()}>
//               Save
//             </button>
//             <button className="btn-sm" onClick={()=>setEditing(false)}>Cancel</button>
//           </div>
//           <p style={{fontSize:10,color:"#555",marginTop:8}}>
//             After saving, you can search for this name in the search bar to find all their photos.
//           </p>
//         </div>
//       )}

//       {data.images===null
//         ? <div className="loader-wrap" style={{minHeight:140}}><div className="loader-ring"/></div>
//         : imgs.length>0
//           ? <ImgGrid list={imgs} onOpen={i=>onOpen(imgs,i)} onFav={()=>{}} onDel={()=>{}}/>
//           : <p className="empty-sub">No photos for this person</p>
//       }
//     </ModalWrap>
//   );
// }

// function AlbumCard({a,onClick,onDelete}){
//   return (
//     <motion.div className="album-card" whileHover={{y:-3,scale:1.01}} onClick={onClick}>
//       {a.cover
//         ? <img src={imgUrl(a.cover)} alt="" className="album-cover" onError={e=>e.target.src=BLANK}/>
//         : <div className="album-cover-ph"><FolderPlus size={32} strokeWidth={1} color="#555"/><span style={{fontSize:10,color:"#555",marginTop:4}}>Empty</span></div>
//       }
//       {a.thumbnails?.length>1&&<div className="album-strip">{a.thumbnails.slice(1,5).map((fn,i)=><img key={i} src={imgUrl(fn)} alt="" className="album-strip-img" onError={e=>e.target.style.display="none"}/>)}</div>}
//       <div className="album-info">
//         <h3 className="album-title">{a.title}</h3>
//         <div className="album-meta">
//           {a.type==="manual"
//             ? <span className="tag-sm tag-sm--manual">✎ manual</span>
//             : <span className="tag-sm tag-sm--accent">📅 event</span>
//           }
//           {/* Only show date badge if it adds info beyond the title (i.e. different month/year span) */}
//           {a.type!=="manual" && a.date && !a.title.includes(a.date.split(" ")[0]) &&
//             <span className="tag-sm">{a.date}</span>
//           }
//           <span className="tag-sm" style={{
//             color: a.count===0 ? "rgba(239,68,68,.7)" : undefined,
//             background: a.count===0 ? "rgba(239,68,68,.08)" : undefined,
//           }}>
//             {a.count===0 ? "⚠ empty" : `📸 ${a.count}`}
//           </span>
//         </div>
//         {a.count===0 && a.type==="manual" && (
//           <p style={{fontSize:9,color:"rgba(96,165,250,.6)",marginTop:4,fontFamily:"var(--mono)"}}>
//             select photos → batch bar → add to album
//           </p>
//         )}
//       </div>
//       {onDelete&&<button className="album-del-btn" onClick={e=>{e.stopPropagation();onDelete();}} title="Delete album"><X size={12}/></button>}
//     </motion.div>
//   );
// }

// function AlbumModal({data,onClose,onOpen,onRename,onDelete}){
//   const imgs=data.images||[];
//   const [editing,setEditing]=React.useState(false);
//   const [title,setTitle]=React.useState(data.title||"");
//   const [desc,setDesc]=React.useState(data.description||"");
//   const saveRename=()=>{ onRename&&onRename(data.id,title,desc); setEditing(false); };
//   return (
//     <ModalWrap onClose={onClose} wide>
//       <div className="modal-head">
//         <div>
//           <h2 className="modal-title">{data.title}</h2>
//           <p className="modal-sub">{data.image_count||data.count||0} photos{data.date?` · ${data.date}`:""}</p>
//         </div>
//         <div style={{display:"flex",gap:6,alignItems:"center"}}>
//           <button className="btn-sm" onClick={()=>setEditing(e=>!e)}>✏ Rename</button>
//           {data.type==="manual"&&onDelete&&(
//             <button className="btn-sm" style={{color:"#f87171",borderColor:"rgba(248,113,113,.3)"}}
//               onClick={()=>{ if(window.confirm(`Delete album "${data.title}"? Photos are kept.`)){onDelete(data.id);onClose();} }}>
//               🗑 Delete
//             </button>
//           )}
//           <button className="lb-hbtn lb-close" onClick={onClose}><X size={17}/></button>
//         </div>
//       </div>
//       {editing&&(
//         <div className="rename-row" style={{marginBottom:12}}>
//           <input className="rename-in" value={title} onChange={e=>setTitle(e.target.value)}
//             onKeyDown={e=>e.key==="Enter"&&saveRename()} placeholder="Album name…" autoFocus/>
//           <input className="rename-in" value={desc} onChange={e=>setDesc(e.target.value)}
//             placeholder="Description (optional)…" style={{marginLeft:6}}/>
//           <button className="btn-sm btn-sm--primary" onClick={saveRename} style={{marginLeft:6}}>Save</button>
//           <button className="btn-sm" onClick={()=>setEditing(false)} style={{marginLeft:4}}>Cancel</button>
//         </div>
//       )}
//       {!editing&&data.description&&(
//         <div className="album-desc"><span className="album-desc-lbl">DESCRIPTION</span><p>{data.description}</p></div>
//       )}
//       {data.images===null
//         ? <div className="loader-wrap" style={{minHeight:140}}><div className="loader-ring"/></div>
//         : imgs.length>0
//           ? <ImgGrid list={imgs} onOpen={i=>onOpen(imgs,i)} onFav={()=>{}} onDel={()=>{}}/>
//           : (
//             <div style={{textAlign:"center",padding:"40px 20px",color:"var(--mu)"}}>
//               <FolderPlus size={40} strokeWidth={1} style={{marginBottom:12,opacity:.4}}/>
//               <p style={{fontSize:14,marginBottom:6}}>This album is empty</p>
//               <p style={{fontSize:12,color:"#444"}}>
//                 Go to <strong>Timeline</strong> → select photos (checkbox icon) →
//                 choose this album in the batch bar → click the layers icon
//               </p>
//             </div>
//           )
//       }
//     </ModalWrap>
//   );
// }

// function ModalWrap({children,onClose,wide}){
//   return <motion.div className="modal-bg" initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}} onClick={onClose}><motion.div className="modal-box" style={{maxWidth:wide?960:680}} initial={{scale:.95,y:12}} animate={{scale:1,y:0}} exit={{scale:.95}} onClick={e=>e.stopPropagation()}>{children}</motion.div></motion.div>;
// }

// function CoOccurrencePanel({faces, onSearch}) {
//   const [selected, setSelected] = React.useState([]);
//   const toggle = id => setSelected(p => p.includes(id) ? p.filter(x=>x!==id) : [...p, id]);
//   return (
//     <div className="create-album-form">
//       <p style={{fontSize:12,color:"var(--mu)",marginBottom:10}}>Select 2+ people to find photos where they all appear together:</p>
//       <div style={{display:"flex",flexWrap:"wrap",gap:8,marginBottom:14,maxHeight:240,overflowY:"auto"}}>
//         {(faces||[]).map(p=>(
//           <button key={p.id}
//             className={"person-chip" + (selected.includes(p.id)?" person-chip--on":"")}
//             onClick={()=>toggle(p.id)}>
//             {p.cover && <img src={imgUrl(p.cover)} alt="" style={{width:24,height:24,borderRadius:"50%",objectFit:"cover",marginRight:6}}/>}
//             {p.name||`Person ${p.id}`}
//           </button>
//         ))}
//         {(!faces||faces.length===0) && <p style={{fontSize:12,color:"var(--mu)"}}>No named people yet. Go to People section first.</p>}
//       </div>
//       <button className="btn-sm btn-sm--primary" disabled={selected.length<2} onClick={()=>onSearch(selected)}>
//         <GitMerge size={12}/> Find Co-appearing Photos ({selected.length} selected)
//       </button>
//     </div>
//   );
// }

// function EmotionTimelineChart() {
//   const [data, setData] = React.useState(null);
//   React.useEffect(()=>{
//     axios.get(`${API}/emotion-timeline`).then(r=>setData(r.data)).catch(()=>{});
//   },[]);

//   if (!data) return <p style={{fontSize:12,color:"var(--mu)",padding:8}}>Loading…</p>;
//   if (!data.months?.length) return (
//     <div style={{padding:"12px 0"}}>
//       <p style={{fontSize:12,color:"var(--mu)"}}>No emotion timeline data yet.</p>
//       <p style={{fontSize:11,color:"#555",marginTop:4}}>
//         Emotions are detected in the background after uploading. Install <code style={{background:"var(--s3)",padding:"1px 5px",borderRadius:3}}>pip install fer</code> for best results, then click <strong>Re-index AI</strong>.
//       </p>
//     </div>
//   );

//   // Show server hint if only neutral detected
//   const serverHint = data.hint;

//   const EMO_COL = {happy:"#4ade80",sad:"#60a5fa",angry:"#f87171",neutral:"#a8a29e",surprised:"#fbbf24",disgusted:"#a78bfa",fearful:"#f9a8d4"};
//   const EMO_EMOJI = {happy:"😊",sad:"😢",angry:"😠",neutral:"😐",surprised:"😲",disgusted:"🤢",fearful:"😨"};

//   const months = data.months;
//   const n = months.length;
//   // Filter to emotions that actually have data (exclude neutral-only if others exist)
//   const showEmos = data.emotions.filter(e => (data.series[e]||[]).some(v=>v>0));
//   const hasRealData = showEmos.some(e => e !== "neutral");

//   // Chart dimensions — extra bottom padding for legend
//   const W=560, CHART_H=160, LEG_H=28, PAD={l:36,r:16,t:12,b:32};
//   const chartW = W - PAD.l - PAD.r;
//   const chartH = CHART_H - PAD.t - PAD.b;
//   const maxVal = Math.max(1, ...showEmos.flatMap(e => data.series[e]||[]));

//   const xScale = i => PAD.l + (n <= 1 ? chartW/2 : (i / (n-1)) * chartW);
//   const yScale = v => PAD.t + chartH - (v / maxVal) * chartH;
//   const makePath = series => series.map((v,i) =>
//     `${i===0?"M":"L"}${xScale(i).toFixed(1)},${yScale(v).toFixed(1)}`
//   ).join(" ");

//   // X labels: show at most 8, evenly spaced
//   const maxLabels = Math.min(8, n);
//   const labelIndices = n <= maxLabels
//     ? months.map((_,i)=>i)
//     : Array.from({length:maxLabels}, (_,k) => Math.round(k*(n-1)/(maxLabels-1)));

//   return (
//     <div style={{overflowX:"auto"}}>
//       {(serverHint || !hasRealData) && (
//         <p style={{fontSize:11,color:"#f59e0b",marginBottom:8,background:"rgba(245,158,11,.08)",padding:"6px 10px",borderRadius:6,border:"1px solid rgba(245,158,11,.2)"}}>
//           ⚠️ {serverHint || 'Only "neutral" detected — run pip install fer then Re-index AI.'}
//         </p>
//       )}
//       <svg viewBox={`0 0 ${W} ${CHART_H + LEG_H}`} style={{width:"100%",maxWidth:W,display:"block"}}>
//         {/* Y grid + labels */}
//         {[0,0.25,0.5,0.75,1].map((f,i)=>{
//           const y = PAD.t + chartH*(1-f);
//           return (
//             <g key={i}>
//               <line x1={PAD.l} x2={W-PAD.r} y1={y} y2={y} stroke="#ffffff08" strokeWidth={1}/>
//               <text x={PAD.l-4} y={y+3} fontSize={7} fill="#555" textAnchor="end">{Math.round(maxVal*f)}</text>
//             </g>
//           );
//         })}
//         {/* Axes */}
//         <line x1={PAD.l} x2={PAD.l} y1={PAD.t} y2={PAD.t+chartH} stroke="#333" strokeWidth={1}/>
//         <line x1={PAD.l} x2={W-PAD.r} y1={PAD.t+chartH} y2={PAD.t+chartH} stroke="#333" strokeWidth={1}/>
//         {/* Emotion lines + dots */}
//         {showEmos.map(e=>(
//           <g key={e}>
//             <path d={makePath(data.series[e]||[])} fill="none" stroke={EMO_COL[e]||"#888"} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round"/>
//             {(data.series[e]||[]).map((v,i)=> v>0 && (
//               <circle key={i} cx={xScale(i)} cy={yScale(v)} r={3} fill={EMO_COL[e]||"#888"} opacity={0.85}/>
//             ))}
//           </g>
//         ))}
//         {/* X axis labels */}
//         {labelIndices.map(i=>(
//           <text key={i} x={xScale(i)} y={PAD.t+chartH+14} fontSize={7} fill="#555" textAnchor="middle">
//             {months[i]}
//           </text>
//         ))}
//         {/* Legend — below chart, wrapping row */}
//         {showEmos.map((e,i)=>{
//           const cols = Math.min(showEmos.length, 4);
//           const lx = PAD.l + (i % cols) * 130;
//           const ly = CHART_H + Math.floor(i/cols)*14 + 4;
//           return (
//             <g key={e} transform={`translate(${lx},${ly})`}>
//               <rect width={8} height={8} fill={EMO_COL[e]||"#888"} rx={2}/>
//               <text x={11} y={7} fontSize={9} fill="#aaa">{EMO_EMOJI[e]} {e}</text>
//             </g>
//           );
//         })}
//       </svg>
//     </div>
//   );
// }

// function PeopleFreqChart({people}) {
//   if(!people?.length) return <p style={{fontSize:12,color:"var(--mu)",padding:8}}>No people data yet.</p>;
//   const top = people.slice(0,12);
//   const maxN = Math.max(...top.map(p=>p.count),1);
//   return (
//     <div style={{display:"flex",flexDirection:"column",gap:6}}>
//       {top.map((p,i)=>(
//         <div key={p.id} style={{display:"flex",alignItems:"center",gap:8}}>
//           <span style={{fontSize:11,color:"var(--mu)",width:20,textAlign:"right"}}>{i+1}</span>
//           <span style={{fontSize:12,color:"var(--tx)",width:120,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{p.name||`Person ${p.id}`}</span>
//           <div style={{flex:1,height:14,background:"var(--s3)",borderRadius:4,overflow:"hidden"}}>
//             <div style={{width:`${(p.count/maxN)*100}%`,height:"100%",background:"linear-gradient(90deg,#818cf8,#a78bfa)",borderRadius:4,transition:"width .4s"}}/>
//           </div>
//           <span style={{fontSize:11,color:"#818cf8",width:30,textAlign:"right"}}>{p.count}</span>
//         </div>
//       ))}
//     </div>
//   );
// }

// function StatsPage({stats, peopleFreq}){
//   if(!stats) return <Empty icon={BarChart3} msg="Loading stats…"/>;
//   const cards=[{l:"Photos",v:stats.total_images,c:"#818cf8"},{l:"Faces",v:stats.total_faces,c:"#a78bfa"},{l:"People",v:stats.total_people,c:"#f472b6"},{l:"Albums",v:stats.total_albums,c:"#fbbf24"},{l:"Favorites",v:stats.total_favorites,c:"#4ade80"},{l:"Indexed",v:stats.indexed_vectors,c:"#22d3ee"}];
//   return (
//     <div className="stats-wrap">
//       <PageHead title="Statistics"/>
//       <div className="stats-cards">{cards.map((c,i)=><motion.div key={i} className="stat-card" whileHover={{y:-3}} style={{"--sc":c.c}}><p className="stat-lbl">{c.l}</p><p className="stat-val" style={{color:c.c}}>{c.v??0}</p></motion.div>)}</div>

//       <div className="stats-sec">
//         <p className="stats-sec-lbl">EMOTION TIMELINE</p>
//         <EmotionTimelineChart/>
//       </div>

//       {peopleFreq?.length>0&&(
//         <div className="stats-sec">
//           <p className="stats-sec-lbl">MOST PHOTOGRAPHED PEOPLE</p>
//           <PeopleFreqChart people={peopleFreq}/>
//         </div>
//       )}

//       {stats.color_distribution?.length>0&&<div className="stats-sec"><p className="stats-sec-lbl">COLOR DISTRIBUTION</p><div className="clr-dist">{stats.color_distribution.map((d,i)=><div key={i} className="clr-dist-item"><div className="clr-dist-dot" style={{background:d.color==="gray"?"#888":d.color==="white"?"#ddd":d.color}}/><span>{d.color}</span><span className="clr-dist-n">×{d.count}</span></div>)}</div></div>}
//       {stats.top_user_tags?.length>0&&<div className="stats-sec"><p className="stats-sec-lbl">YOUR TAGS</p><div className="tag-cloud">{stats.top_user_tags.map((t,i)=><span key={i} className="tag-cloud-item" style={{borderColor:"#86efac44",color:"#86efac"}}>#{t.tag}<span className="tag-n" style={{color:"#86efac99"}}> ×{t.count}</span></span>)}</div></div>}
//       {stats.top_tags?.length>0&&<div className="stats-sec"><p className="stats-sec-lbl">AI DETECTED OBJECTS</p><div className="tag-cloud">{stats.top_tags.map((t,i)=><span key={i} className="tag-cloud-item">{t.tag}<span className="tag-n"> ×{t.count}</span></span>)}</div></div>}
//     </div>
//   );
// }

// function ToolsAccordion({reindexing, onEmotions, onColors, onNames, onCaptions, onRecaptionAll, onCleanup}) {
//   const [open, setOpen] = React.useState(false);
//   return (
//     <div className="tools-accordion">
//       <button className="tools-accordion-btn" onClick={()=>setOpen(p=>!p)}>
//         <span style={{display:"flex",alignItems:"center",gap:6}}>
//           <SlidersHorizontal size={11}/> AI Tools
//         </span>
//         <motion.span animate={{rotate:open?180:0}} transition={{duration:.2}}>
//           <ChevronDown size={11}/>
//         </motion.span>
//       </button>
//       <AnimatePresence>
//         {open && (
//           <motion.div initial={{height:0,opacity:0}} animate={{height:"auto",opacity:1}} exit={{height:0,opacity:0}}
//             style={{overflow:"hidden"}}>
//             <div style={{display:"flex",flexDirection:"column",gap:4,paddingTop:4}}>
//               {[
//                 {label:"Fix Captions",    color:"rgba(167,139,250,.9)", bg:"rgba(167,139,250,.07)", border:"rgba(167,139,250,.2)", fn:onCaptions,     title:"Caption images that are missing descriptions"},
//                 {label:"Re-caption All",  color:"rgba(129,140,248,.85)",bg:"rgba(129,140,248,.07)", border:"rgba(129,140,248,.2)", fn:onRecaptionAll, title:"Re-run BLIP on every image (slow — use after model upgrade)"},
//                 {label:"Fix Emotions",    color:"rgba(250,204,21,.8)",  bg:"rgba(250,204,21,.07)",  border:"rgba(250,204,21,.2)",  fn:onEmotions,     title:"Re-detect emotions on all photos"},
//                 {label:"Fix Colors",      color:"rgba(52,211,153,.85)", bg:"rgba(52,211,153,.07)",  border:"rgba(52,211,153,.2)",  fn:onColors,       title:"Recompute dominant colors for color search"},
//                 {label:"Auto-Name",       color:"rgba(96,165,250,.85)", bg:"rgba(96,165,250,.07)",  border:"rgba(96,165,250,.2)",  fn:onNames,        title:"Auto-detect names from photo captions"},
//                 {label:"Clean Albums",    color:"rgba(248,113,113,.8)", bg:"rgba(248,113,113,.07)", border:"rgba(248,113,113,.2)", fn:onCleanup,      title:"Delete empty auto-generated albums"},
//               ].map(({label,color,bg,border,fn,title})=>(
//                 <button key={label} disabled={reindexing}
//                   className="reindex-btn"
//                   title={title}
//                   style={{background:bg,borderColor:border,color,fontSize:10,padding:"7px"}}
//                   onClick={fn}>
//                   {label}
//                 </button>
//               ))}
//             </div>
//           </motion.div>
//         )}
//       </AnimatePresence>
//     </div>
//   );
// }

// function PageHead({title,count,unit="photos",extra}){ return <div className="page-head"><h1 className="page-title">{title}</h1>{count>0&&<span className="page-count">{count} {unit}</span>}{extra&&<div style={{marginLeft:"auto"}}>{extra}</div>}</div>; }
// function Empty({icon:Icon,msg,sub,onClick}){ return <div className={`empty ${onClick?"empty--click":""}`} onClick={onClick}><Icon size={44} strokeWidth={1} color="#333"/><p className="empty-msg">{msg}</p>{sub&&<p className="empty-sub">{sub}</p>}{onClick&&<p className="empty-cta">CLICK TO UPLOAD</p>}</div>; }
// function Fade({children}){ return <motion.div initial={{opacity:0,y:6}} animate={{opacity:1,y:0}} exit={{opacity:0}} transition={{duration:.18}}>{children}</motion.div>; }

// /* ══════════════════════════════════════════════════════════════════════════ */
// /* CSS                                                                         */
// /* ══════════════════════════════════════════════════════════════════════════ */
// function CSS() {
//   return <style>{`
// @import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300..900;1,14..32,300..900&family=JetBrains+Mono:wght@400;500&display=swap');

// *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

// :root{
//   /* Background layers */
//   --bg0:#000000;
//   --bg1:#0a0a0a;
//   --bg2:#111111;
//   --bg3:#161616;
//   --bg4:#1c1c1c;

//   /* Borders */
//   --b1:rgba(255,255,255,.06);
//   --b2:rgba(255,255,255,.1);
//   --b3:rgba(255,255,255,.15);

//   /* Accent */
//   --ac:#6366f1;
//   --ac-l:#818cf8;
//   --ac-d:#4f46e5;
//   --ac-glow:rgba(99,102,241,.25);
//   --ac-bg:rgba(99,102,241,.08);

//   /* Text */
//   --t1:#ffffff;
//   --t2:#a1a1aa;
//   --t3:#52525b;

//   /* Status */
//   --green:#22c55e;
//   --red:#ef4444;
//   --gold:#f59e0b;
//   --sky:#38bdf8;
//   --rose:#f43f5e;

//   --font:'Inter',system-ui,sans-serif;
//   --mono:'JetBrains Mono',monospace;
//   --r:8px;
//   --r2:12px;
//   --r3:16px;
// }

// html,body,#root{height:100%;background:var(--bg0);color:var(--t1);font-family:var(--font);overflow:hidden;-webkit-font-smoothing:antialiased}

// ::-webkit-scrollbar{width:3px;height:3px}
// ::-webkit-scrollbar-track{background:transparent}
// ::-webkit-scrollbar-thumb{background:var(--b2);border-radius:3px}

// /* ═══ LAYOUT ══════════════════════════════════════════════════════════════ */
// .shell{display:flex;height:100vh;overflow:hidden;background:var(--bg0)}

// /* ═══ SIDEBAR ═════════════════════════════════════════════════════════════ */
// .sidebar{
//   width:220px;flex-shrink:0;
//   background:var(--bg1);
//   border-right:1px solid var(--b1);
//   display:flex;flex-direction:column;
//   padding:0;overflow-y:auto;overflow-x:hidden;
// }

// /* Brand */
// .brand{
//   display:flex;align-items:center;gap:10px;
//   padding:20px 16px 16px;
//   border-bottom:1px solid var(--b1);
//   flex-shrink:0;
// }
// .brand-mark{
//   width:32px;height:32px;border-radius:8px;
//   background:var(--ac);
//   display:flex;align-items:center;justify-content:center;
//   flex-shrink:0;color:#fff;
//   box-shadow:0 0 0 1px rgba(99,102,241,.4),0 4px 12px rgba(99,102,241,.3);
// }
// .brand-name{font-size:14px;font-weight:700;letter-spacing:-.01em;color:var(--t1)}
// .brand-sub{font-family:var(--mono);font-size:9px;color:var(--t3);letter-spacing:.06em;margin-top:1px}

// /* Nav */
// .nav-list{display:flex;flex-direction:column;padding:8px;gap:1px;flex-shrink:0}
// .nav-btn{
//   display:flex;align-items:center;gap:9px;
//   width:100%;padding:8px 10px;
//   border-radius:var(--r);background:transparent;border:none;
//   font-family:var(--font);font-size:13px;font-weight:500;
//   color:var(--t3);cursor:pointer;text-align:left;
//   transition:color .12s,background .12s;position:relative;
// }
// .nav-btn:hover{background:var(--bg3);color:var(--t2)}
// .nav-btn--on{background:var(--bg3);color:var(--t1);font-weight:600}
// .nav-btn--on .nav-ic{color:var(--ac-l)}
// .nav-btn--trash:hover{color:var(--red);background:rgba(239,68,68,.06)}
// .nav-ic{display:flex;flex-shrink:0;color:inherit}
// .nav-pip{
//   position:absolute;right:8px;
//   width:4px;height:4px;border-radius:50%;background:var(--ac);
//   box-shadow:0 0 6px var(--ac);
// }
// .badge{
//   margin-left:auto;background:var(--red);color:#fff;
//   font-size:10px;font-family:var(--mono);
//   padding:1px 5px;border-radius:20px;line-height:1.5;
// }

// .sidebar-sep{height:1px;background:var(--b1);margin:6px 16px;flex-shrink:0}
// .sidebar-section-label{
//   display:flex;align-items:center;gap:5px;
//   font-size:10px;font-weight:600;color:var(--t3);
//   letter-spacing:.06em;text-transform:uppercase;
//   padding:8px 16px 4px;flex-shrink:0;
// }

// /* Tag nav */
// .tag-nav-list{display:flex;flex-direction:column;padding:0 8px;gap:1px}
// .tag-nav-btn{
//   display:flex;align-items:center;justify-content:space-between;
//   width:100%;padding:6px 10px;border-radius:var(--r);
//   background:transparent;border:none;
//   font-family:var(--mono);font-size:11px;color:var(--t3);
//   cursor:pointer;transition:all .12s;
// }
// .tag-nav-btn:hover{background:var(--bg3);color:var(--green)}
// .tag-nav-btn--on{background:rgba(34,197,94,.07);color:var(--green)}
// .tag-nav-count{font-size:9px;color:var(--t3)}

// /* Sidebar footer */
// .sidebar-foot{
//   display:flex;flex-direction:column;gap:4px;
//   margin-top:auto;padding:12px 8px;border-top:1px solid var(--b1);
//   flex-shrink:0;
// }
// .reindex-btn{
//   display:flex;align-items:center;justify-content:center;gap:6px;
//   background:transparent;border:1px solid var(--b1);
//   border-radius:var(--r);padding:8px;
//   font-family:var(--font);font-size:12px;font-weight:500;
//   color:var(--t3);cursor:pointer;transition:all .12s;
// }
// .reindex-btn:hover{border-color:var(--b2);color:var(--t2);background:var(--bg3)}
// .reindex-btn:disabled{opacity:.35;cursor:default}
// .reindex-btn--primary{
//   background:var(--ac-bg);border-color:rgba(99,102,241,.3);
//   color:var(--ac-l);font-weight:600;
// }
// .reindex-btn--primary:hover{background:rgba(99,102,241,.15);border-color:var(--ac)}

// .reindex-msg{font-family:var(--mono);font-size:9px;color:var(--t3);text-align:center;padding:2px 4px;line-height:1.5}

// .offline-pill{
//   display:flex;align-items:center;gap:6px;
//   background:rgba(34,197,94,.05);border:1px solid rgba(34,197,94,.12);
//   border-radius:var(--r);padding:7px 10px;
//   font-size:11px;font-weight:500;color:rgba(34,197,94,.7);
// }
// .dot-green{width:6px;height:6px;border-radius:50%;background:var(--green);box-shadow:0 0 6px var(--green);flex-shrink:0}

// /* Tools accordion */
// .tools-accordion{border:1px solid var(--b1);border-radius:var(--r);overflow:hidden;background:var(--bg2)}
// .tools-accordion-btn{
//   display:flex;align-items:center;justify-content:space-between;
//   width:100%;padding:8px 10px;background:transparent;border:none;
//   font-family:var(--font);font-size:11px;font-weight:500;color:var(--t3);cursor:pointer;
//   transition:color .12s;
// }
// .tools-accordion-btn:hover{color:var(--t2)}

// /* ═══ MAIN ════════════════════════════════════════════════════════════════ */
// .main{flex:1;display:flex;flex-direction:column;min-width:0;overflow:hidden}

// /* ═══ TOPBAR ══════════════════════════════════════════════════════════════ */
// .topbar{
//   background:var(--bg1);
//   border-bottom:1px solid var(--b1);
//   padding:10px 20px;
//   display:flex;flex-direction:column;gap:8px;flex-shrink:0;
// }

// /* Mode pills */
// .mode-strip{display:flex;gap:2px;align-items:center;flex-wrap:wrap}
// .mode-btn{
//   display:flex;align-items:center;gap:5px;
//   padding:5px 10px;border-radius:6px;border:none;cursor:pointer;
//   font-family:var(--font);font-size:11px;font-weight:500;
//   background:transparent;color:var(--t3);transition:all .12s;
// }
// .mode-btn:hover{background:var(--bg3);color:var(--t2)}
// .mode-btn--on{background:var(--ac);color:#fff;font-weight:600}
// .voice-chip{
//   margin-left:6px;display:flex;align-items:center;gap:4px;
//   font-family:var(--mono);font-size:9px;letter-spacing:.05em;
//   padding:4px 10px;border-radius:6px;
// }
// .voice-chip--rec{background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.2);color:var(--red)}
// .voice-chip--processing{background:var(--ac-bg);border:1px solid rgba(99,102,241,.2);color:var(--ac-l)}
// .voice-chip--error{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.15);color:#f87171}

// /* Search bar */
// .search-bar{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
// .search-wrap{
//   flex:1;min-width:200px;
//   display:flex;align-items:center;gap:8px;
//   background:var(--bg2);border:1px solid var(--b1);
//   border-radius:var(--r2);padding:0 12px;
//   transition:border-color .15s;
// }
// .search-wrap:focus-within{border-color:rgba(99,102,241,.4)}
// .search-ic{color:var(--t3);flex-shrink:0}
// .search-in{
//   flex:1;background:transparent;border:none;
//   padding:9px 0;font-family:var(--font);font-size:13px;
//   color:var(--t1);outline:none;
// }
// .search-in::placeholder{color:var(--t3)}
// .mic-btn{background:transparent;border:none;cursor:pointer;color:var(--t3);padding:3px;display:flex;border-radius:5px;transition:color .12s}
// .mic-btn:hover{color:var(--t2)}
// .mic-btn--rec{color:var(--red)}
// .mic-btn--proc{color:var(--ac-l)}

// .img-search-label{
//   display:flex;align-items:center;gap:7px;cursor:pointer;
//   background:var(--bg2);border:1px solid var(--b1);
//   border-radius:var(--r2);padding:8px 12px;
//   font-size:12px;color:var(--t3);flex-shrink:0;transition:all .12s;
// }
// .img-search-label:hover{border-color:var(--b2);color:var(--t2)}
// .img-search-thumb{width:20px;height:20px;border-radius:4px;object-fit:cover}

// .color-strip{display:flex;gap:5px;align-items:center;flex-wrap:wrap}
// .clr-dot{
//   width:22px;height:22px;border-radius:50%;border:none;cursor:pointer;
//   outline:2px solid transparent;outline-offset:2px;transition:all .12s;flex-shrink:0;
// }
// .clr-dot:hover{transform:scale(1.15)}
// .clr-dot--on{outline-color:rgba(255,255,255,.5);transform:scale(1.2)}

// .btn-go{
//   background:var(--ac);color:#fff;border:none;border-radius:var(--r2);
//   padding:8px 16px;font-family:var(--font);font-size:12px;font-weight:600;
//   cursor:pointer;transition:background .12s;flex-shrink:0;
// }
// .btn-go:hover{background:var(--ac-d)}

// .btn-batch{
//   display:flex;align-items:center;gap:5px;
//   background:var(--bg2);border:1px solid var(--b1);
//   color:var(--t3);border-radius:var(--r2);
//   padding:8px 12px;font-family:var(--font);font-size:12px;font-weight:500;
//   cursor:pointer;transition:all .12s;flex-shrink:0;
// }
// .btn-batch:hover{border-color:var(--b2);color:var(--t2)}
// .btn-batch--on{background:var(--ac-bg);border-color:rgba(99,102,241,.3);color:var(--ac-l)}

// .btn-upload{
//   display:flex;align-items:center;gap:5px;
//   background:var(--bg2);border:1px solid var(--b2);
//   color:var(--t2);border-radius:var(--r2);
//   padding:8px 14px;font-family:var(--font);font-size:12px;font-weight:600;
//   cursor:pointer;transition:all .12s;flex-shrink:0;
// }
// .btn-upload:hover{border-color:var(--b3);color:var(--t1);background:var(--bg3)}

// /* Emotion filter bar */
// .emo-filter-bar{display:flex;align-items:center;gap:5px;flex-wrap:wrap}
// .emo-filter-label{font-family:var(--mono);font-size:9px;color:var(--t3);letter-spacing:.06em;flex-shrink:0}
// .emo-filter-btn{
//   display:flex;align-items:center;gap:4px;
//   padding:4px 10px;border-radius:20px;
//   border:1px solid var(--b1);background:transparent;
//   color:var(--t3);font-family:var(--font);font-size:11px;font-weight:500;
//   cursor:pointer;transition:all .12s;flex-shrink:0;
// }
// .emo-filter-btn:hover{border-color:var(--b2);color:var(--t2)}
// .emo-filter-count{font-family:var(--mono);font-size:8px;opacity:.5;margin-left:2px}
// .emo-filter-clear{
//   display:flex;align-items:center;gap:4px;padding:4px 9px;
//   border-radius:20px;border:1px solid rgba(239,68,68,.2);
//   background:rgba(239,68,68,.06);color:var(--red);
//   font-family:var(--font);font-size:11px;font-weight:500;cursor:pointer;transition:all .12s;
// }

// /* Emoji bar */
// .emoji-quick-bar{display:flex;align-items:center;gap:3px;flex-wrap:wrap}
// .emoji-quick-label{font-family:var(--mono);font-size:9px;color:var(--t3);letter-spacing:.06em;flex-shrink:0;margin-right:2px}
// .emoji-quick-btn{
//   background:var(--bg3);border:1px solid var(--b1);
//   border-radius:6px;padding:3px 6px;font-size:14px;
//   cursor:pointer;transition:all .12s;line-height:1;
// }
// .emoji-quick-btn:hover{background:var(--ac-bg);border-color:rgba(99,102,241,.3);transform:scale(1.12)}

// /* ═══ BODY ════════════════════════════════════════════════════════════════ */
// .body{flex:1;overflow-y:auto;padding:20px 20px 80px}
// .page-head{display:flex;align-items:baseline;gap:10px;margin-bottom:18px;flex-wrap:wrap}
// .page-title{font-size:18px;font-weight:700;letter-spacing:-.025em;color:var(--t1)}
// .page-count{font-family:var(--mono);font-size:9px;color:var(--t3);letter-spacing:.05em}
// .sel-count{font-family:var(--mono);font-size:9px;color:var(--ac-l);background:var(--ac-bg);padding:2px 8px;border-radius:20px;border:1px solid rgba(99,102,241,.2)}

// /* On This Day */
// .otd-banner{
//   background:rgba(245,158,11,.04);
//   border:1px solid rgba(245,158,11,.12);
//   border-radius:var(--r3);margin-bottom:18px;overflow:hidden;
// }
// .otd-header{display:flex;align-items:center;justify-content:space-between;padding:12px 16px;cursor:pointer}
// .otd-left{display:flex;align-items:center;gap:10px}
// .otd-title{font-size:13px;font-weight:600;display:block}
// .otd-sub{font-family:var(--mono);font-size:9px;color:rgba(245,158,11,.5);display:block;margin-top:1px}
// .otd-toggle{background:none;border:none;cursor:pointer;color:var(--t3);display:flex;padding:3px;border-radius:5px}
// .otd-year{padding:0 16px 14px}
// .otd-year-label{font-family:var(--mono);font-size:9px;color:rgba(245,158,11,.5);letter-spacing:.06em;margin-bottom:8px;display:block}
// .otd-strip{display:flex;gap:8px;overflow-x:auto;padding-bottom:4px}
// .otd-thumb{flex-shrink:0;width:96px;cursor:pointer;border-radius:var(--r);overflow:hidden;border:1px solid var(--b1);transition:border-color .15s}
// .otd-thumb:hover{border-color:rgba(245,158,11,.3)}
// .otd-thumb img{width:100%;height:64px;object-fit:cover;display:block}
// .otd-thumb-caption{font-size:9px;color:var(--t3);padding:4px 6px;line-height:1.4;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}

// /* ═══ IMAGE GRID ══════════════════════════════════════════════════════════ */
// .img-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(176px,1fr));gap:8px}
// .img-grid--sm{grid-template-columns:repeat(auto-fill,minmax(124px,1fr));gap:6px}

// /* Photo card */
// .photo-card{
//   position:relative;aspect-ratio:1;
//   border-radius:var(--r2);overflow:hidden;cursor:pointer;
//   background:var(--bg2);border:1px solid var(--b1);
//   transition:border-color .15s;
// }
// .photo-card:hover{border-color:var(--b2)}
// .photo-card--sel{border-color:var(--ac)!important;box-shadow:0 0 0 2px rgba(99,102,241,.3)!important}
// .photo-img{width:100%;height:100%;object-fit:cover;display:block;transition:transform .25s}
// .photo-card:hover .photo-img{transform:scale(1.04)}
// .chip{position:absolute;font-family:var(--mono);font-size:8px;border-radius:4px;border:1px solid;padding:1px 5px;pointer-events:none;backdrop-filter:blur(4px)}
// .chip--q{top:7px;left:7px}
// .chip--emo{top:7px;right:7px;font-size:13px;border:none;background:none;padding:0}
// .fav-dot{position:absolute;bottom:7px;right:7px;color:var(--rose);font-size:13px;pointer-events:none}
// .batch-check{position:absolute;top:7px;left:7px}
// .photo-hover{
//   position:absolute;inset:0;opacity:0;transition:opacity .18s;
//   background:linear-gradient(to top,rgba(0,0,0,.9) 0%,transparent 55%);
//   display:flex;flex-direction:column;justify-content:flex-end;
// }
// .photo-card:hover .photo-hover{opacity:1}
// .photo-actions{position:absolute;top:7px;right:7px;display:flex;gap:4px}
// .ph-btn{
//   width:28px;height:28px;border-radius:6px;border:none;cursor:pointer;
//   display:flex;align-items:center;justify-content:center;transition:all .12s;
//   backdrop-filter:blur(6px);
// }
// .ph-btn--fav{background:rgba(244,63,94,.7);color:#fff}
// .ph-btn--del{background:rgba(0,0,0,.6);border:1px solid rgba(255,255,255,.1);color:#888}
// .ph-btn--del:hover{background:rgba(239,68,68,.8);color:#fff;border-color:transparent}
// .photo-info{padding:8px 8px 6px}
// .photo-caption{font-size:10px;line-height:1.45;color:rgba(255,255,255,.9);margin-bottom:4px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
// .photo-meta-row{display:flex;flex-wrap:wrap;gap:3px}
// .meta-chip{font-family:var(--mono);font-size:7.5px;color:rgba(255,255,255,.45);background:rgba(0,0,0,.4);padding:1px 5px;border-radius:3px;text-transform:capitalize}

// /* ═══ BATCH BAR ═══════════════════════════════════════════════════════════ */
// .batch-bar{
//   position:absolute;bottom:0;left:0;right:0;
//   background:rgba(10,10,10,.97);backdrop-filter:blur(12px);
//   border-top:1px solid var(--b1);
//   padding:10px 20px;
//   display:flex;align-items:center;gap:10px;flex-wrap:wrap;z-index:50;
// }
// .batch-bar-left{display:flex;align-items:center;gap:7px}
// .batch-count{display:flex;align-items:center;gap:5px;font-size:12px;font-weight:600;color:var(--ac-l)}
// .batch-sm{background:var(--bg3);border:1px solid var(--b1);color:var(--t3);border-radius:6px;padding:4px 10px;font-family:var(--font);font-size:11px;font-weight:500;cursor:pointer;transition:all .12s}
// .batch-sm:hover{border-color:var(--b2);color:var(--t2)}
// .batch-bar-actions{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-left:auto}
// .batch-action-btn{
//   display:flex;align-items:center;gap:5px;padding:6px 12px;
//   border-radius:var(--r);border:1px solid var(--b1);
//   background:var(--bg2);color:var(--t2);
//   font-family:var(--font);font-size:11px;font-weight:500;cursor:pointer;transition:all .12s;
// }
// .batch-action-btn:hover{border-color:var(--b2);color:var(--t1);background:var(--bg3)}
// .batch-action-btn--fav:hover{border-color:rgba(244,63,94,.3);color:var(--rose)}
// .batch-action-btn--del:hover{border-color:rgba(239,68,68,.3);color:var(--red)}
// .batch-action-btn--album{border-color:rgba(56,189,248,.2);color:var(--sky)}
// .batch-action-btn--album:hover{background:rgba(56,189,248,.08);border-color:rgba(56,189,248,.3)}
// .batch-input-group{display:flex;gap:3px;align-items:center}
// .batch-input{background:var(--bg2);border:1px solid var(--b1);border-radius:var(--r);padding:6px 10px;font-family:var(--font);font-size:11px;color:var(--t1);outline:none;width:108px;transition:border-color .12s}
// .batch-input:focus{border-color:rgba(99,102,241,.4)}
// .batch-select{background:var(--bg2);border:1px solid var(--b1);border-radius:var(--r);padding:6px 10px;font-family:var(--font);font-size:11px;color:var(--t2);outline:none;cursor:pointer}
// .batch-select:focus{border-color:rgba(99,102,241,.4)}

// /* ═══ LIGHTBOX ════════════════════════════════════════════════════════════ */
// .lb-bg{
//   position:fixed;inset:0;z-index:300;
//   background:rgba(0,0,0,.9);backdrop-filter:blur(16px);
//   display:flex;align-items:center;justify-content:center;padding:14px;
// }
// .lb-box{
//   width:100%;max-width:1100px;max-height:92vh;
//   background:var(--bg1);border:1px solid var(--b1);
//   border-radius:16px;display:grid;grid-template-columns:1fr 300px;
//   overflow:hidden;box-shadow:0 24px 80px rgba(0,0,0,.8);
// }
// .lb-left{position:relative;background:var(--bg0);display:flex;align-items:center;justify-content:center;min-height:380px;overflow:hidden}
// .lb-img{max-width:100%;max-height:90vh;object-fit:contain;display:block}
// .lb-nav{
//   position:absolute;top:50%;transform:translateY(-50%);
//   width:38px;height:38px;border-radius:50%;border:1px solid var(--b2);
//   cursor:pointer;background:rgba(0,0,0,.5);backdrop-filter:blur(6px);
//   color:#fff;display:flex;align-items:center;justify-content:center;
//   transition:all .15s;z-index:2;
// }
// .lb-nav:hover{background:var(--ac);border-color:transparent}
// .lb-nav--l{left:12px}
// .lb-nav--r{right:12px}
// .lb-counter{position:absolute;bottom:12px;left:50%;transform:translateX(-50%);font-family:var(--mono);font-size:9px;color:rgba(255,255,255,.35);background:rgba(0,0,0,.5);padding:2px 10px;border-radius:20px}
// .lb-hint{position:absolute;bottom:12px;right:14px;font-family:var(--mono);font-size:7.5px;color:rgba(255,255,255,.15);letter-spacing:.05em}
// .lb-right{border-left:1px solid var(--b1);background:var(--bg1);display:flex;flex-direction:column;min-height:0}
// .lb-right-head{display:flex;align-items:center;justify-content:space-between;padding:14px 16px;border-bottom:1px solid var(--b1);flex-shrink:0}
// .lb-right-title{font-size:13px;font-weight:600;letter-spacing:-.01em}
// .lb-head-btns{display:flex;align-items:center;gap:3px}
// .lb-hbtn{background:none;border:none;cursor:pointer;color:var(--t3);padding:6px;border-radius:6px;display:flex;align-items:center;transition:all .12s}
// .lb-hbtn:hover{background:var(--bg3);color:var(--t2)}
// .lb-close:hover{color:var(--red)}
// .lb-scroll{flex:1;overflow-y:auto;padding:12px 14px;display:flex;flex-direction:column;gap:7px}

// /* Detail blocks */
// .db{background:var(--bg2);border:1px solid var(--b1);border-radius:var(--r2);overflow:hidden;border-left:2px solid var(--dc,var(--ac))}
// .db-head{display:flex;align-items:center;gap:6px;padding:7px 11px;border-bottom:1px solid var(--b1)}
// .db-label{font-family:var(--mono);font-size:8px;color:var(--dc,var(--ac-l));letter-spacing:.08em;font-weight:500;text-transform:uppercase}
// .db-body{padding:9px 11px}
// .db-caption{font-size:12px;line-height:1.55;color:var(--t1)}
// .db-caption-sub{font-size:11px;line-height:1.55;color:var(--t2);margin-top:4px}
// .q-wrap{display:flex;align-items:flex-start;gap:10px}
// .q-circle{width:42px;height:42px;border-radius:50%;border:2px solid;display:flex;align-items:center;justify-content:center;flex-shrink:0}
// .q-right{flex:1}
// .q-level{font-size:12px;font-weight:600;margin-bottom:6px}
// .qbar-row{display:flex;align-items:center;gap:7px;margin-bottom:4px}
// .qbar-label{font-family:var(--mono);font-size:7.5px;color:var(--t3);width:65px;flex-shrink:0}
// .qbar-track{flex:1;height:2px;background:rgba(255,255,255,.07);border-radius:2px;overflow:hidden}
// .qbar-fill{height:100%;border-radius:2px}
// .qbar-num{font-family:var(--mono);font-size:7.5px;color:var(--t3);width:22px;text-align:right}
// .aes-row{display:flex;align-items:center;gap:9px}
// .aes-score{font-size:24px;font-weight:800;color:var(--gold)}
// .aes-rating{font-size:11px;color:var(--t2)}
// .emo-row{display:flex;align-items:center;gap:10px}
// .emo-name{font-size:12px;font-weight:600;text-transform:capitalize}
// .emo-sub{font-family:var(--mono);font-size:8px;color:var(--t3);margin-top:2px}
// .ocr-text{font-family:var(--mono);font-size:9px;color:var(--t2);line-height:1.6;white-space:pre-wrap;max-height:84px;overflow-y:auto}
// .tag-wrap{display:flex;flex-wrap:wrap;gap:4px}
// .obj-tag{font-family:var(--mono);font-size:8.5px;background:var(--bg3);color:var(--t3);padding:2px 7px;border-radius:4px;border:1px solid var(--b1)}
// .meta-stack{display:flex;flex-direction:column;gap:5px}
// .mrow{display:flex;justify-content:space-between;align-items:baseline;gap:12px}
// .mrow-k{font-family:var(--mono);font-size:8px;color:var(--t3);flex-shrink:0}
// .mrow-v{font-size:11px;color:var(--t1);text-align:right}

// /* Tags */
// .tag-chips{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px}
// .tag-chip-item{display:flex;align-items:center;gap:3px;background:rgba(34,197,94,.07);border:1px solid rgba(34,197,94,.15);color:var(--green);font-family:var(--mono);font-size:8.5px;padding:2px 7px;border-radius:4px}
// .tag-chip-x{background:none;border:none;cursor:pointer;color:rgba(34,197,94,.4);display:flex;padding:0;margin-left:2px;transition:color .12s}
// .tag-chip-x:hover{color:var(--green)}
// .tag-add-row{display:flex;gap:4px;align-items:center}
// .tag-add-input{flex:1;background:var(--bg3);border:1px solid var(--b1);border-radius:6px;padding:5px 9px;font-family:var(--mono);font-size:10px;color:var(--t1);outline:none;transition:border-color .12s}
// .tag-add-input:focus{border-color:rgba(99,102,241,.4)}
// .tag-add-btn{background:var(--bg3);border:1px solid var(--b1);border-radius:6px;padding:5px 8px;cursor:pointer;color:var(--t3);display:flex;transition:all .12s}
// .tag-add-btn:hover{border-color:rgba(99,102,241,.3);color:var(--ac-l)}

// /* ═══ TRASH ═══════════════════════════════════════════════════════════════ */
// .trash-card{position:relative;aspect-ratio:1;border-radius:var(--r2);overflow:hidden;background:var(--bg2);border:1px solid var(--b1)}
// .trash-img{width:100%;height:100%;object-fit:cover;filter:grayscale(.5) brightness(.4)}
// .trash-overlay{position:absolute;inset:0;opacity:0;transition:opacity .18s;background:rgba(0,0,0,.7);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:6px;padding:10px}
// .trash-card:hover .trash-overlay{opacity:1}
// .trash-btn{width:100%;display:flex;align-items:center;justify-content:center;gap:5px;border:none;border-radius:var(--r);padding:7px;font-family:var(--font);font-size:11px;font-weight:600;cursor:pointer;transition:all .12s}
// .trash-btn--restore{background:var(--ac);color:#fff}
// .trash-btn--del{background:rgba(220,50,50,.85);color:#fff}
// .trash-date{position:absolute;top:7px;left:7px;font-family:var(--mono);font-size:7.5px;background:rgba(0,0,0,.7);color:var(--t3);padding:2px 6px;border-radius:4px}

// /* ═══ PEOPLE ══════════════════════════════════════════════════════════════ */
// .people-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(108px,1fr));gap:18px}
// .person-card{display:flex;flex-direction:column;align-items:center;gap:8px;cursor:pointer}
// .person-avatar{width:68px;height:68px;border-radius:50%;overflow:hidden;border:1px solid var(--b1);background:var(--bg2);display:flex;align-items:center;justify-content:center;transition:border-color .15s}
// .person-avatar img{width:100%;height:100%;object-fit:cover}
// .person-card:hover .person-avatar{border-color:var(--ac)}
// .person-name{font-size:12px;font-weight:600;text-align:center}
// .person-count{font-family:var(--mono);font-size:8px;color:var(--t3)}

// /* ═══ ALBUMS ══════════════════════════════════════════════════════════════ */
// .albums-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px}
// .album-card{
//   position:relative;aspect-ratio:16/10;
//   border-radius:var(--r3);overflow:hidden;cursor:pointer;
//   background:var(--bg2);border:1px solid var(--b1);
//   transition:border-color .15s;
// }
// .album-card:hover{border-color:var(--b2)}
// .album-cover{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;transition:transform .3s}
// .album-card:hover .album-cover{transform:scale(1.04)}
// .album-cover-ph{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;color:var(--t3)}
// .album-strip{position:absolute;bottom:0;left:0;right:0;height:28px;display:flex;opacity:0;transition:opacity .18s;z-index:2}
// .album-card:hover .album-strip{opacity:1}
// .album-strip-img{flex:1;height:100%;object-fit:cover;border-right:1px solid rgba(0,0,0,.3)}
// .album-info{position:absolute;inset:0;z-index:1;background:linear-gradient(to top,rgba(0,0,0,.9) 0%,transparent 55%);padding:12px;display:flex;flex-direction:column;justify-content:flex-end}
// .album-title{font-weight:700;font-size:13px;letter-spacing:-.01em;margin-bottom:5px;line-height:1.3;color:#fff}
// .album-meta{display:flex;align-items:center;gap:5px;flex-wrap:wrap}
// .album-del-btn{position:absolute;top:8px;right:8px;width:24px;height:24px;border-radius:50%;background:rgba(0,0,0,.65);border:1px solid var(--b2);color:var(--t3);display:flex;align-items:center;justify-content:center;cursor:pointer;z-index:3;opacity:0;transition:all .15s}
// .album-card:hover .album-del-btn{opacity:1}
// .album-del-btn:hover{background:rgba(220,50,50,.85);color:#fff;border-color:transparent}
// .tag-sm{font-family:var(--mono);font-size:8px;background:rgba(255,255,255,.08);color:var(--t3);padding:2px 6px;border-radius:20px}
// .tag-sm--accent{background:rgba(99,102,241,.15);color:var(--ac-l)}
// .tag-sm--manual{background:rgba(245,158,11,.1);color:rgba(245,158,11,.8)}

// /* ═══ DUPLICATES ══════════════════════════════════════════════════════════ */
// .dupe-list{display:flex;flex-direction:column;gap:14px}
// .dupe-group{background:var(--bg2);border:1px solid var(--b1);border-radius:var(--r3);padding:16px}
// .dupe-head{display:flex;align-items:center;gap:7px;margin-bottom:14px;flex-wrap:wrap}

// /* ═══ MODAL ═══════════════════════════════════════════════════════════════ */
// .modal-bg{position:fixed;inset:0;z-index:200;background:rgba(0,0,0,.8);backdrop-filter:blur(10px);display:flex;align-items:center;justify-content:center;padding:18px}
// .modal-box{width:100%;max-height:90vh;overflow-y:auto;background:var(--bg1);border:1px solid var(--b1);border-radius:var(--r3);padding:20px;box-shadow:0 16px 50px rgba(0,0,0,.7)}
// .modal-head{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:14px;gap:10px}
// .modal-title{font-size:18px;font-weight:700;letter-spacing:-.025em}
// .modal-sub{font-family:var(--mono);font-size:8px;color:var(--t3);letter-spacing:.06em;margin-top:3px}
// .rename-row{display:flex;gap:7px;margin-bottom:14px;align-items:center}
// .rename-in{flex:1;background:var(--bg2);border:1px solid var(--b1);border-radius:var(--r);padding:8px 11px;font-family:var(--font);font-size:13px;color:var(--t1);outline:none;transition:border-color .12s}
// .rename-in:focus{border-color:rgba(99,102,241,.4)}
// .album-desc{background:var(--ac-bg);border:1px solid rgba(99,102,241,.15);border-radius:var(--r);padding:10px 12px;margin-bottom:14px}
// .album-desc-lbl{font-family:var(--mono);font-size:8px;color:var(--ac-l);letter-spacing:.08em;display:block;margin-bottom:4px}
// .album-desc p{font-size:12px;line-height:1.6}
// .create-album-form{display:flex;flex-direction:column;gap:9px}
// .form-input{background:var(--bg2);border:1px solid var(--b1);border-radius:var(--r);padding:10px 12px;font-family:var(--font);font-size:13px;color:var(--t1);outline:none;transition:border-color .12s}
// .form-input:focus{border-color:rgba(99,102,241,.4)}
// .form-textarea{background:var(--bg2);border:1px solid var(--b1);border-radius:var(--r);padding:10px 12px;font-family:var(--font);font-size:13px;color:var(--t1);outline:none;resize:vertical;transition:border-color .12s}
// .form-textarea:focus{border-color:rgba(99,102,241,.4)}

// .btn-sm{display:inline-flex;align-items:center;gap:5px;background:var(--bg2);border:1px solid var(--b1);color:var(--t2);border-radius:var(--r);padding:6px 12px;font-family:var(--font);font-size:11px;font-weight:500;cursor:pointer;transition:all .12s}
// .btn-sm:hover{border-color:var(--b2);color:var(--t1);background:var(--bg3)}
// .btn-sm--primary{background:var(--ac);border-color:transparent;color:#fff}
// .btn-sm--primary:hover{background:var(--ac-d)}
// .btn-sm--accent{background:var(--ac-bg);border-color:rgba(99,102,241,.25);color:var(--ac-l)}
// .btn-sm--accent:hover{background:var(--ac);color:#fff;border-color:transparent}
// .btn-sm:disabled{opacity:.35;cursor:default}

// /* ═══ STATS ═══════════════════════════════════════════════════════════════ */
// .stats-wrap{max-width:820px}
// .stats-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:14px}
// .stat-card{
//   background:var(--bg2);border:1px solid var(--b1);
//   border-radius:var(--r2);padding:16px 18px;
//   border-top:2px solid var(--sc);
//   transition:border-color .15s;
// }
// .stat-card:hover{border-color:var(--b2)}
// .stat-lbl{font-family:var(--mono);font-size:8px;color:var(--t3);letter-spacing:.09em;margin-bottom:8px;text-transform:uppercase}
// .stat-val{font-size:36px;font-weight:800;letter-spacing:-.04em;line-height:1;color:var(--sc)}
// .stats-sec{background:var(--bg2);border:1px solid var(--b1);border-radius:var(--r2);padding:14px;margin-bottom:8px}
// .stats-sec-lbl{font-family:var(--mono);font-size:8px;color:var(--t3);letter-spacing:.09em;margin-bottom:12px;text-transform:uppercase}
// .clr-dist{display:flex;flex-wrap:wrap;gap:6px}
// .clr-dist-item{display:flex;align-items:center;gap:6px;background:var(--bg3);border:1px solid var(--b1);border-radius:6px;padding:5px 10px;font-size:11px}
// .clr-dist-dot{width:10px;height:10px;border-radius:50%;border:1px solid rgba(255,255,255,.1);flex-shrink:0}
// .clr-dist-n{font-family:var(--mono);font-size:9px;color:var(--t3)}
// .tag-cloud{display:flex;flex-wrap:wrap;gap:5px}
// .tag-cloud-item{font-family:var(--mono);font-size:9px;background:var(--ac-bg);color:var(--ac-l);border:1px solid rgba(99,102,241,.15);padding:3px 9px;border-radius:20px}
// .tag-n{opacity:.5}

// /* ═══ EMPTY / LOADER ══════════════════════════════════════════════════════ */
// .empty{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:240px;background:var(--bg2);border:1px dashed var(--b1);border-radius:var(--r3);padding:40px;gap:10px;text-align:center}
// .empty--click{cursor:pointer;transition:background .12s}
// .empty--click:hover{background:var(--bg3)}
// .empty-msg{font-size:16px;font-weight:700;letter-spacing:-.02em}
// .empty-sub{font-size:12px;color:var(--t2);max-width:240px;line-height:1.6}
// .empty-cta{font-family:var(--mono);font-size:9px;color:var(--ac-l);letter-spacing:.09em;margin-top:4px;text-transform:uppercase}
// .loader-wrap{display:flex;flex-direction:column;align-items:center;justify-content:center;height:55vh;gap:12px}
// .loader-ring{width:34px;height:34px;border-radius:50%;border:2px solid rgba(99,102,241,.12);border-top-color:var(--ac);animation:spin .7s linear infinite}
// .loader-text{font-family:var(--mono);font-size:10px;color:var(--t3);letter-spacing:.1em}
// @keyframes spin{to{transform:rotate(360deg)}}
// .spin{animation:spin .7s linear infinite}

// /* ═══ MISC ════════════════════════════════════════════════════════════════ */
// .person-chip{display:flex;align-items:center;padding:5px 11px;background:var(--bg2);border:1px solid var(--b1);border-radius:20px;font-size:12px;color:var(--t3);cursor:pointer;transition:all .12s}
// .person-chip:hover{border-color:rgba(99,102,241,.3);color:var(--t2)}
// .person-chip--on{background:rgba(49,46,129,.4);border-color:var(--ac);color:var(--ac-l)}

// @media(max-width:720px){
//   .lb-box{grid-template-columns:1fr}
//   .lb-right{border-left:none;border-top:1px solid var(--b1);max-height:40vh}
//   .stats-cards{grid-template-columns:repeat(2,1fr)}
//   .sidebar{width:180px}
// }
// `}</style>;
// }

// // import React, { useState, useEffect, useRef, useCallback } from "react";
// // import { motion, AnimatePresence } from "framer-motion";
// // import {
// //   Search, Upload, Image as ImgIcon, Clock, BookOpen, Users, Heart,
// //   Copy, Trash2, X, ChevronLeft, ChevronRight, Sparkles, BarChart3,
// //   GitMerge, Palette, Shuffle, RotateCcw, AlertTriangle, FileText,
// //   Mic, MicOff, Zap, Star, Eye, Camera, Info, Tag, Type, Smile,
// //   Loader2, Trash, BookImage, ImageOff, TrendingUp, Aperture,
// //   CheckSquare, Square, Plus, Calendar, Hash, FolderPlus,
// //   CheckCheck, Layers, ChevronDown, ChevronUp,
// //   SlidersHorizontal, UserCheck
// // } from "lucide-react";
// // import axios from "axios";

// // const API = "http://localhost:8000";
// // const imgUrl = (f) => {
// //   if (!f) return null;
// //   const b = String(f).split("/").filter(Boolean).pop();
// //   return b ? `${API}/image/${b}` : null;
// // };
// // const BLANK = `data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='1' height='1'%3E%3Crect fill='%23111118'/%3E%3C/svg%3E`;

// // const QL = {
// //   Excellent: { c:"#4ade80", bg:"rgba(74,222,128,.12)" },
// //   Good:      { c:"#818cf8", bg:"rgba(129,140,248,.12)" },
// //   Fair:      { c:"#fbbf24", bg:"rgba(251,191,36,.12)"  },
// //   Poor:      { c:"#f87171", bg:"rgba(248,113,113,.12)" },
// // };
// // const EMO = { happy:"😊",sad:"😢",angry:"😠",surprised:"😲",disgusted:"🤢",fearful:"😨",neutral:"😐" };
// // const EMO_COLORS = { happy:"#fbbf24",sad:"#60a5fa",angry:"#f87171",surprised:"#a78bfa",disgusted:"#4ade80",fearful:"#fb923c",neutral:"#6b7280" };
// // const qInfo = (l) => QL[l] || { c:"#555", bg:"rgba(255,255,255,.05)" };

// // /* ─── Voice hook ─────────────────────────────────────────────────────────── */
// // function useVoice(onResult, onError) {
// //   const [state, setState] = useState("idle");
// //   const streamRef = useRef(null);
// //   const processorRef = useRef(null);
// //   const ctxRef = useRef(null);
// //   const samplesRef = useRef([]);
// //   const ok = typeof navigator !== "undefined" && !!navigator.mediaDevices?.getUserMedia;

// //   // Convert Float32 PCM samples → 16-bit WAV blob (no ffmpeg needed)
// //   const buildWav = (samples, sampleRate) => {
// //     const len = samples.length;
// //     const buf = new ArrayBuffer(44 + len * 2);
// //     const view = new DataView(buf);
// //     const write = (off, str) => { for(let i=0;i<str.length;i++) view.setUint8(off+i, str.charCodeAt(i)); };
// //     write(0,"RIFF"); view.setUint32(4, 36+len*2, true);
// //     write(8,"WAVE"); write(12,"fmt ");
// //     view.setUint32(16,16,true); view.setUint16(20,1,true); view.setUint16(22,1,true);
// //     view.setUint32(24,sampleRate,true); view.setUint32(28,sampleRate*2,true);
// //     view.setUint16(32,2,true); view.setUint16(34,16,true);
// //     write(36,"data"); view.setUint32(40,len*2,true);
// //     for(let i=0;i<len;i++){
// //       const s = Math.max(-1,Math.min(1,samples[i]));
// //       view.setInt16(44+i*2, s<0?s*0x8000:s*0x7FFF, true);
// //     }
// //     return new Blob([buf], {type:"audio/wav"});
// //   };

// //   const start = useCallback(async () => {
// //     if (!ok) return onError?.("Mic unavailable");
// //     try {
// //       const stream = await navigator.mediaDevices.getUserMedia({
// //         audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true }
// //       });
// //       streamRef.current = stream;
// //       samplesRef.current = [];

// //       // Use Web Audio API to capture raw PCM at 16kHz — bypasses ffmpeg completely
// //       const ctx = new AudioContext({ sampleRate: 16000 });
// //       ctxRef.current = ctx;
// //       const source = ctx.createMediaStreamSource(stream);
// //       const processor = ctx.createScriptProcessor(4096, 1, 1);
// //       processorRef.current = processor;

// //       processor.onaudioprocess = e => {
// //         const data = e.inputBuffer.getChannelData(0);
// //         samplesRef.current.push(new Float32Array(data));
// //       };
// //       source.connect(processor);
// //       processor.connect(ctx.destination);
// //       setState("rec");
// //     } catch(e) {
// //       setState("error");
// //       onError?.(e.name === "NotAllowedError" ? "Mic permission denied" : "Mic unavailable");
// //       setTimeout(()=>setState("idle"),3000);
// //     }
// //   }, [ok]);

// //   const stop = useCallback(async () => {
// //     if (state !== "rec") return;
// //     setState("processing");
// //     try {
// //       // Stop all tracks
// //       streamRef.current?.getTracks().forEach(t=>t.stop());
// //       processorRef.current?.disconnect();
// //       await ctxRef.current?.close();

// //       // Flatten all captured chunks into one array
// //       const allSamples = samplesRef.current;
// //       if (!allSamples.length) {
// //         setState("error"); onError?.("No audio captured"); setTimeout(()=>setState("idle"),3000); return;
// //       }
// //       const total = allSamples.reduce((s,c)=>s+c.length,0);
// //       const merged = new Float32Array(total);
// //       let off=0; for(const c of allSamples){merged.set(c,off);off+=c.length;}

// //       // Build WAV and send to Vosk (no ffmpeg needed — already 16kHz mono PCM)
// //       const wav = buildWav(merged, 16000);
// //       const fd = new FormData(); fd.append("audio", wav, "rec.wav");
// //       const {data} = await axios.post(`${API}/voice_search`, fd, {timeout:30000});
// //       if (data.success && data.transcript) { setState("idle"); onResult(data.transcript); }
// //       else { setState("error"); onError?.(data.error||"No speech detected — speak clearly and try again"); setTimeout(()=>setState("idle"),4000); }
// //     } catch(err) {
// //       const detail = err?.response?.data?.detail || err.message || "";
// //       setState("error");
// //       onError?.(err?.response?.status===503 ? "Vosk model not loaded" : detail.slice(0,60) || "Voice failed");
// //       setTimeout(()=>setState("idle"),5000);
// //     }
// //   }, [state]);

// //   return { state, ok, start, stop };
// // }

// // /* ══════════════════════════════════════════════════════════════════════════ */
// // export default function App() {
// //   const [view,        setView]        = useState("timeline");
// //   const [images,      setImages]      = useState([]);
// //   const [faces,       setFaces]       = useState([]);
// //   const [albums,      setAlbums]      = useState([]);
// //   const [dupes,       setDupes]       = useState([]);
// //   const [trash,       setTrash]       = useState([]);
// //   const [stats,       setStats]       = useState(null);
// //   const [loading,     setLoading]     = useState(false);
// //   const [searchQ,     setSearchQ]     = useState("");
// //   const [searchMode,  setSearchMode]  = useState("text");
// //   const [colorPick,   setColorPick]   = useState("blue");
// //   const [hybridFile,  setHybridFile]  = useState(null);
// //   const [hybridPrev,  setHybridPrev]  = useState(null);
// //   const [voiceErr,    setVoiceErr]    = useState("");
// //   const [reindexMsg,  setReindexMsg]  = useState("");
// //   const [reindexing,  setReindexing]  = useState(false);

// //   /* New feature state */
// //   const [emoSummary,    setEmoSummary]    = useState([]);
// //   const [emoFilter,     setEmoFilter]     = useState("");        // active emotion filter
// //   const [onThisDay,     setOnThisDay]     = useState(null);
// //   const [otdExpanded,   setOtdExpanded]   = useState(false);

// //   const [batchMode,     setBatchMode]     = useState(false);
// //   const [selected,      setSelected]      = useState(new Set());
// //   const [batchTagInput, setBatchTagInput] = useState("");
// //   const [batchAlbumId,  setBatchAlbumId]  = useState("");

// //   const [allTags,       setAllTags]       = useState([]);
// //   const [activeTag,     setActiveTag]     = useState("");

// //   const [createAlbumOpen, setCreateAlbumOpen] = useState(false);
// //   const [newAlbumTitle,   setNewAlbumTitle]   = useState("");
// //   const [newAlbumDesc,    setNewAlbumDesc]    = useState("");
// //   const [pendingAlbumImages, setPendingAlbumImages] = useState([]); // snapshot of selected at open time

// //   // Named event creation
// //   const [createEventOpen, setCreateEventOpen] = useState(false);
// //   const [newEventTitle,   setNewEventTitle]   = useState("");
// //   const [newEventType,    setNewEventType]    = useState("Other");
// //   const [newEventDate,    setNewEventDate]    = useState("");
// //   const [newEventDesc,    setNewEventDesc]    = useState("");

// //   // Advanced filters
// //   const [minPeople,     setMinPeople]     = useState(0);
// //   const [coPersonIds,   setCoPersonIds]   = useState([]);
// //   const [showCoModal,   setShowCoModal]   = useState(false);

// //   // Face similarity
// //   const [faceSrcFile,   setFaceSrcFile]   = useState(null);
// //   const [faceSrcPrev,   setFaceSrcPrev]   = useState(null);
// //   const [showFaceSim,   setShowFaceSim]   = useState(false);

// //   // People frequency
// //   const [peopleFreq,    setPeopleFreq]    = useState([]);
// //   // People suggestions shown when search returns no results
// //   const [peopleSuggestions, setPeopleSuggestions] = useState([]);

// //   /* Lightbox */
// //   const [lb, setLb] = useState(null);
// //   /* Modals */
// //   const [personModal, setPersonModal] = useState(null);
// //   const [albumModal,  setAlbumModal]  = useState(null);

// //   const uploadRef = useRef(null);
// //   const voice = useVoice(
// //     t => { setSearchQ(t); setSearchMode("text"); setVoiceErr(""); runSearch("text",t); },
// //     setVoiceErr
// //   );

// //   /* keyboard */
// //   useEffect(() => {
// //     const h = e => {
// //       if (!lb) return;
// //       if (e.key==="ArrowRight") shiftLb(1);
// //       if (e.key==="ArrowLeft")  shiftLb(-1);
// //       if (e.key==="Escape")     setLb(null);
// //     };
// //     window.addEventListener("keydown",h);
// //     return ()=>window.removeEventListener("keydown",h);
// //   }, [lb]);

// //   const shiftLb = useCallback(d=>setLb(p=>p?{...p,idx:(p.idx+d+p.list.length)%p.list.length}:null),[]);
// //   const openLb  = (list,idx)=>setLb({list,idx});
// //   const closeLb = ()=>setLb(null);

// //   /* load */
// //   useEffect(()=>{ load(); },[view]);
// //   useEffect(()=>{
// //     loadEmoSummary(); loadTags(); loadTrashCount(); loadPeopleFreq();

// //   }, []);
// //   useEffect(()=>{ if(view==="timeline") loadOnThisDay(); }, [view]);

// //   const load = async () => {
// //     setLoading(true);
// //     try {
// //       if (view==="timeline")   { const r=await axios.get(`${API}/timeline`);              setImages(r.data.results||[]); }
// //       if (view==="favorites")  { const r=await axios.get(`${API}/favorites`);             setImages(r.data.results||[]); }
// //       if (view==="explore")    { const r=await axios.get(`${API}/explore/random?count=24`);setImages(r.data.results||[]); }
// //       if (view==="faces")      { const r=await axios.get(`${API}/faces`);                 setFaces(r.data.results||[]); }
// //       if (view==="albums")     { const r=await axios.get(`${API}/albums`);                setAlbums(r.data.results||[]); }
// //       if (view==="duplicates") { const r=await axios.get(`${API}/duplicates`);            setDupes(r.data.duplicate_groups||[]); }
// //       if (view==="trash")      { const r=await axios.get(`${API}/trash`);                 setTrash(r.data.results||[]); }
// //       if (view==="stats")      { const r=await axios.get(`${API}/stats`);                 setStats(r.data); }
// //     } catch {}
// //     setSelected(new Set()); setBatchMode(false);
// //     setLoading(false);
// //   };

// //   const loadEmoSummary = async () => {
// //     try { const r=await axios.get(`${API}/emotions/summary`); setEmoSummary(r.data.emotions||[]); } catch {}
// //   };
// //   const loadOnThisDay = async () => {
// //     try { const r=await axios.get(`${API}/on-this-day`); setOnThisDay(r.data); } catch {}
// //   };
// //   const loadTags = async () => {
// //     try { const r=await axios.get(`${API}/tags`); setAllTags(r.data.tags||[]); } catch {}
// //   };
// //   const loadTrashCount = async () => {
// //     try { const r=await axios.get(`${API}/trash`); setTrash(r.data.results||[]); } catch {}
// //   };

// //   const loadPeopleFreq = async () => {
// //     try { const r=await axios.get(`${API}/people/frequency`); setPeopleFreq(r.data.people||[]); } catch {}
// //   };

// //   // Group photo filter
// //   const filterGroupPhotos = async (n) => {
// //     setMinPeople(n); setLoading(true); setView("search");
// //     try { const r=await axios.get(`${API}/group-photos?min_people=${n}`); setImages(r.data.results||[]); }
// //     catch {} finally { setLoading(false); }
// //   };

// //   // Co-occurrence search
// //   const runCoOccurrence = async (ids) => {
// //     if (!ids||ids.length<2) return;
// //     setLoading(true); setView("search"); setShowCoModal(false);
// //     try { const r=await axios.get(`${API}/co-occurrence?person_ids=${ids.join(",")}`); setImages(r.data.results||[]); }
// //     catch {} finally { setLoading(false); }
// //   };

// //   // Face similarity
// //   const runFaceSimilarity = async () => {
// //     if (!faceSrcFile) return;
// //     setLoading(true); setView("search"); setShowFaceSim(false);
// //     const fd=new FormData(); fd.append("file", faceSrcFile); fd.append("top_k","30");
// //     try { const r=await axios.post(`${API}/face-similarity`, fd); setImages(r.data.results||[]); }
// //     catch(err) { alert("Face search failed: "+(err?.response?.data?.detail||err.message)); }
// //     finally { setLoading(false); setFaceSrcFile(null); setFaceSrcPrev(null); }
// //   };

// //   const runSearch = async (mode=searchMode, q=searchQ) => {
// //     setEmoFilter(""); setActiveTag(""); setLoading(true); setView("search");
// //     try {
// //       const fd=new FormData(); let res;
// //       if (mode==="text")     { fd.append("query",q);          res=await axios.post(`${API}/search`,fd); }
// //       if (mode==="describe") { fd.append("description",q);    res=await axios.post(`${API}/search/describe`,fd); }
// //       if (mode==="image")    { fd.append("file",hybridFile); fd.append("top_k",20); res=await axios.post(`${API}/search/image`,fd); }
// //       if (mode==="hybrid")   { fd.append("query",q); if(hybridFile)fd.append("file",hybridFile); fd.append("top_k",20); res=await axios.post(`${API}/search/hybrid`,fd); }
// //       if (mode==="color")    { fd.append("color",colorPick); fd.append("top_k",20); res=await axios.post(`${API}/search/color`,fd); }
// //       let results = res?.data?.results || [];
// //       // If text search returned nothing, try person name search as fallback
// //       if (mode==="text" && results.length===0 && q.trim().length>=2) {
// //         try {
// //           const pr = await axios.get(`${API}/people/search?q=${encodeURIComponent(q.trim())}`);
// //           if (pr.data.results?.length) {
// //             results = pr.data.results.flatMap(p => p.results || []);
// //           }
// //         } catch {}
// //       }
// //       // Show people suggestions if still no results
// //       if (mode==="text" && results.length===0) {
// //         const suggestions = res?.data?.people_suggestions || [];
// //         setPeopleSuggestions(suggestions);
// //       } else {
// //         setPeopleSuggestions([]);
// //       }
// //       setImages(results);
// //     } catch { setImages([]); }
// //     setLoading(false);
// //   };

// //   const filterByEmotion = async (emo) => {
// //     if (emoFilter===emo) { setEmoFilter(""); load(); return; }
// //     setEmoFilter(emo); setView("search"); setLoading(true);
// //     try { const fd=new FormData(); fd.append("emotion",emo); const r=await axios.post(`${API}/search/emotion`,fd); setImages(r.data.results||[]); }
// //     catch { setImages([]); }
// //     setLoading(false);
// //   };

// //   const filterByTag = async (tag) => {
// //     if (activeTag===tag) { setActiveTag(""); load(); return; }
// //     setActiveTag(tag); setView("timeline"); setLoading(true);
// //     try { const r=await axios.get(`${API}/tags/${tag}/images`); setImages(r.data.results||[]); }
// //     catch { setImages([]); }
// //     setLoading(false);
// //   };

// //   const handleUpload = async (files) => {
// //     setLoading(true);
// //     for (const f of Array.from(files)) { try { const fd=new FormData(); fd.append("file",f); await axios.post(`${API}/upload`,fd); } catch {} }
// //     load();
// //   };

// //   const doReindex = async () => {
// //     setReindexing(true); setReindexMsg("Clustering…");
// //     try { const r=await axios.post(`${API}/recluster`); setReindexMsg(`✓ ${r.data.people} people · ${r.data.albums} albums`); setTimeout(load,500); }
// //     catch { setReindexMsg("Failed"); }
// //     setReindexing(false);
// //   };

// //   const reprocessEmotions = async () => {
// //     setReindexing(true); setReindexMsg("Reprocessing emotions…");
// //     try {
// //       const r = await axios.post(`${API}/reprocess-emotions`);
// //       setReindexMsg(`✓ ${r.data.message}`);
// //       setTimeout(()=>{ loadEmoSummary(); if(view==="stats") load(); }, 3000);
// //     } catch(err) {
// //       setReindexMsg("Emotion reprocess failed: " + (err?.response?.data?.detail || err.message));
// //     }
// //     setReindexing(false);
// //   };

// //   const reprocessColors = async () => {
// //     setReindexing(true); setReindexMsg("Recomputing colors…");
// //     try {
// //       const r = await axios.post(`${API}/reprocess-colors`);
// //       setReindexMsg(`✓ Colors updated for ${r.data.updated} photos`);
// //     } catch(err) {
// //       setReindexMsg("Color reprocess failed: " + (err?.response?.data?.detail || err.message));
// //     }
// //     setReindexing(false);
// //   };

// //   const reprocessNames = async () => {
// //     setReindexing(true); setReindexMsg("Auto-naming people…");
// //     try {
// //       const r = await axios.post(`${API}/reprocess-names`);
// //       setReindexMsg(`✓ ${r.data.message}`);
// //       setTimeout(()=>{ if(view==="faces") load(); }, 1000);
// //     } catch(err) {
// //       setReindexMsg("Name detection failed: " + (err?.response?.data?.detail || err.message));
// //     }
// //     setReindexing(false);
// //   };

// //   const toggleFav = async (id, e) => {
// //     e?.stopPropagation();
// //     try { const fd=new FormData(); fd.append("image_id",id); await axios.post(`${API}/toggle_favorite`,fd); if(view==="favorites")load(); } catch {}
// //   };

// //   const softDelete = async (id, e) => {
// //     e?.stopPropagation();
// //     if (!window.confirm("Move to trash?")) return;
// //     try {
// //       const fd=new FormData(); fd.append("image_id",id);
// //       await axios.post(`${API}/delete_image`,fd);
// //       setImages(p=>p.filter(i=>i.id!==id));
// //       loadTrashCount();
// //     } catch(err) {
// //       alert("Could not move to trash: " + (err?.response?.data?.detail || err.message));
// //     }
// //   };

// //   const restoreImg = async id => {
// //     try {
// //       const fd=new FormData(); fd.append("image_id",id);
// //       await axios.post(`${API}/restore`,fd);
// //       setTrash(p=>p.filter(i=>i.id!==id));
// //     } catch(err) {
// //       alert("Restore failed: " + (err?.response?.data?.detail || err.message));
// //     }
// //   };
// //   const permDelete = async id => {
// //     if (!window.confirm("Delete forever? This cannot be undone.")) return;
// //     try {
// //       const fd=new FormData(); fd.append("image_id",id);
// //       await axios.post(`${API}/permanent_delete`,fd);
// //       setTrash(p=>p.filter(i=>i.id!==id));
// //     } catch(err) {
// //       alert("Delete failed: " + (err?.response?.data?.detail || err.message));
// //     }
// //   };

// //   const openPerson = async p => {
// //     setPersonModal({...p,images:null});
// //     try { const r=await axios.get(`${API}/people/${p.id}`); setPersonModal(r.data); } catch {}
// //   };
// //   const openAlbum = async a => {
// //     setAlbumModal({...a,images:null});
// //     try { const r=await axios.get(`${API}/albums/${a.id}`); setAlbumModal(r.data); } catch {}
// //   };
// //   const renamePerson = async (id,name) => {
// //     try { const fd=new FormData(); fd.append("name",name); await axios.post(`${API}/people/${id}`,fd); load(); setPersonModal(p=>p&&{...p,name}); } catch {}
// //   };

// //   /* Batch actions */
// //   const toggleSelect = useCallback((id) => {
// //     setSelected(prev => { const s=new Set(prev); s.has(id)?s.delete(id):s.add(id); return s; });
// //   }, []);
// //   const selectAll = () => setSelected(new Set(images.map(i=>i.id)));
// //   const clearSel  = () => setSelected(new Set());

// //   const batchFavorite = async (val) => {
// //     if (!selected.size) return;
// //     const fd=new FormData(); fd.append("image_ids",[...selected].join(",")); fd.append("value",val?1:0);
// //     try { await axios.post(`${API}/batch/favorite`,fd); load(); } catch {}
// //   };
// //   const batchDelete = async () => {
// //     if (!selected.size||!window.confirm(`Trash ${selected.size} photos?`)) return;
// //     const fd=new FormData(); fd.append("image_ids",[...selected].join(","));
// //     try { await axios.post(`${API}/batch/delete`,fd); load(); } catch {}
// //   };
// //   const batchTag = async () => {
// //     if (!selected.size||!batchTagInput.trim()) return;
// //     const fd=new FormData(); fd.append("image_ids",[...selected].join(",")); fd.append("tag",batchTagInput.trim());
// //     try { await axios.post(`${API}/batch/tag`,fd); setBatchTagInput(""); loadTags(); } catch {}
// //   };
// //   const batchAddAlbum = async () => {
// //     if (!selected.size||!batchAlbumId) return;
// //     const fd=new FormData(); fd.append("image_ids",[...selected].join(",")); fd.append("album_id",batchAlbumId);
// //     try { await axios.post(`${API}/batch/album`,fd); setBatchAlbumId(""); load(); } catch {}
// //   };

// //   const createAlbum = async (imageIds=[]) => {
// //     if (!newAlbumTitle.trim()) return;
// //     const fd=new FormData();
// //     fd.append("title", newAlbumTitle);
// //     fd.append("description", newAlbumDesc);
// //     // Priority: explicit imageIds arg → pendingAlbumImages snapshot → current selected
// //     const ids = imageIds.length ? imageIds
// //               : pendingAlbumImages.length ? pendingAlbumImages
// //               : [...selected];
// //     console.log("Creating album with", ids.length, "images:", ids);
// //     if (ids.length) fd.append("image_ids", ids.join(","));
// //     try {
// //       const r = await axios.post(`${API}/albums/create`, fd);
// //       setCreateAlbumOpen(false); setNewAlbumTitle(""); setNewAlbumDesc("");
// //       const addedCount = ids.length;
// //       setPendingAlbumImages([]);
// //       setSelected(new Set()); setBatchMode(false);
// //       setView("albums");
// //       await load();
// //       // Open the newly created album immediately so user can see it
// //       if (r.data?.id) {
// //         const newAlbum = {id: r.data.id, title: r.data.title, type:"manual",
// //                          count: addedCount, images: null};
// //         openAlbum(newAlbum);
// //       }
// //       return r.data;
// //     } catch(err) { alert("Failed to create album: " + (err?.response?.data?.detail || err.message)); }
// //   };
// //   const deleteAlbum = async id => {
// //     try {
// //       await axios.delete(`${API}/albums/${id}/delete`);
// //       setAlbumModal(null);
// //       load();
// //     } catch(err) { alert("Delete failed: " + (err?.response?.data?.detail || err.message)); }
// //   };

// //   const renameAlbum = async (id, title, description) => {
// //     const fd=new FormData(); fd.append("title", title);
// //     if (description) fd.append("description", description);
// //     try {
// //       await axios.post(`${API}/albums/${id}/rename`, fd);
// //       load();
// //       setAlbumModal(prev => prev && prev.id===id ? {...prev, title, description: description||prev.description} : prev);
// //     } catch(err) { alert("Rename failed: " + (err?.response?.data?.detail || err.message)); }
// //   };

// //   const EVENT_TYPES = ["Birthday","Trip","Vacation","Wedding","Anniversary","Graduation","Party","Holiday","Family","Work","Other"];

// //   const createNamedEvent = async () => {
// //     if (!newEventTitle.trim()) return;
// //     const fd=new FormData();
// //     fd.append("title", newEventTitle.trim());
// //     fd.append("event_type", newEventType);
// //     fd.append("description", newEventDesc.trim());
// //     fd.append("date_str", newEventDate);
// //     try {
// //       await axios.post(`${API}/events/create`, fd);
// //       setCreateEventOpen(false);
// //       setNewEventTitle(""); setNewEventType("Other"); setNewEventDate(""); setNewEventDesc("");
// //       load();
// //     } catch(err) { alert("Could not create event: " + (err?.response?.data?.detail || err.message)); }
// //   };

// //   const addTagToImage = async (imgId, tag) => {
// //     const fd=new FormData(); fd.append("tag",tag);
// //     try {
// //       const r = await axios.post(`${API}/photo/${imgId}/tags/add`,fd);
// //       const newTags = r.data.tags || [];
// //       // Update the images list in place so lightbox reflects latest tags
// //       setImages(prev => prev.map(i => i.id===imgId ? {...i, user_tags: newTags} : i));
// //       loadTags();
// //       return newTags;
// //     } catch { return null; }
// //   };
// //   const removeTagFromImage = async (imgId, tag) => {
// //     const fd=new FormData(); fd.append("tag",tag);
// //     try {
// //       const r = await axios.post(`${API}/photo/${imgId}/tags/remove`,fd);
// //       const newTags = r.data.tags || [];
// //       setImages(prev => prev.map(i => i.id===imgId ? {...i, user_tags: newTags} : i));
// //       loadTags();
// //       return newTags;
// //     } catch { return null; }
// //   };

// //   const isGridView = ["timeline","search","favorites","explore"].includes(view);

// //   const NAV = [
// //     { id:"timeline",   icon:<Clock size={15}/>,     label:"Timeline"   },
// //     { id:"albums",     icon:<BookOpen size={15}/>,  label:"Albums"     },
// //     { id:"faces",      icon:<Users size={15}/>,     label:"People"     },
// //     { id:"favorites",  icon:<Heart size={15}/>,     label:"Favorites"  },
// //     { id:"duplicates", icon:<Copy size={15}/>,      label:"Duplicates" },
// //     { id:"explore",    icon:<Shuffle size={15}/>,   label:"Explore"    },
// //     { id:"stats",      icon:<BarChart3 size={15}/>, label:"Statistics" },
// //   ];
// //   const MODES = [
// //     { id:"text",     Icon:Search,   label:"Search"   },
// //     { id:"describe", Icon:FileText, label:"Describe" },
// //     { id:"image",    Icon:ImgIcon,  label:"By Image" },
// //     { id:"hybrid",   Icon:GitMerge, label:"Hybrid"   },
// //     { id:"color",    Icon:Palette,  label:"By Color" },
// //   ];
// //   const COLORS = ["red","orange","yellow","green","blue","purple","pink","white","black","gray","brown"];

// //   return (
// //     <>
// //       <CSS />
// //       <div className="shell">

// //         {/* ── SIDEBAR ─────────────────────────────────────────────────── */}
// //         <nav className="sidebar">
// //           <div className="brand">
// //             <div className="brand-mark"><Aperture size={16} strokeWidth={1.5}/></div>
// //             <div>
// //               <p className="brand-name">LUMINA</p>
// //               <p className="brand-sub">AI Gallery</p>
// //             </div>
// //           </div>

// //           <div className="nav-list">
// //             {NAV.map(n=>(
// //               <button key={n.id} className={`nav-btn ${view===n.id?"nav-btn--on":""}`} onClick={()=>setView(n.id)}>
// //                 <span className="nav-ic">{n.icon}</span>
// //                 <span>{n.label}</span>
// //                 {view===n.id && <span className="nav-pip"/>}
// //               </button>
// //             ))}
// //           </div>

// //           <div className="sidebar-sep"/>

// //           {/* Advanced Filters */}
// //           <p className="sidebar-section-label"><SlidersHorizontal size={10}/> FILTERS</p>
// //           <button className="nav-btn" style={{fontSize:11}} onClick={()=>filterGroupPhotos(2)}>
// //             <span className="nav-ic"><Users size={14}/></span><span>2+ People</span>
// //           </button>
// //           <button className="nav-btn" style={{fontSize:11}} onClick={()=>filterGroupPhotos(3)}>
// //             <span className="nav-ic"><Users size={14}/></span><span>3+ People</span>
// //           </button>
// //           <button className="nav-btn" style={{fontSize:11}} onClick={()=>filterGroupPhotos(5)}>
// //             <span className="nav-ic"><Users size={14}/></span><span>5+ People</span>
// //           </button>
// //           <button className="nav-btn" style={{fontSize:11}} onClick={()=>setShowCoModal(true)}>
// //             <span className="nav-ic"><GitMerge size={14}/></span><span>Co-Appear</span>
// //           </button>
// //           <button className="nav-btn" style={{fontSize:11}} onClick={()=>setShowFaceSim(true)}>
// //             <span className="nav-ic"><UserCheck size={14}/></span><span>Face Search</span>
// //           </button>

// //           <div className="sidebar-sep"/>

// //           <button className={`nav-btn nav-btn--trash ${view==="trash"?"nav-btn--on":""}`} onClick={()=>setView("trash")}>
// //             <span className="nav-ic"><Trash size={15}/></span>
// //             <span>Trash</span>
// //             {trash.length>0 && <span className="badge">{trash.length}</span>}
// //           </button>

// //           {/* Tags sidebar section */}
// //           {allTags.length>0 && (
// //             <>
// //               <div className="sidebar-sep"/>
// //               <p className="sidebar-section-label"><Hash size={10}/> TAGS</p>
// //               <div className="tag-nav-list">
// //                 {allTags.slice(0,8).map(t=>(
// //                   <button key={t.tag} className={`tag-nav-btn ${activeTag===t.tag?"tag-nav-btn--on":""}`} onClick={()=>filterByTag(t.tag)}>
// //                     <span>#{t.tag}</span><span className="tag-nav-count">{t.count}</span>
// //                   </button>
// //                 ))}
// //               </div>
// //             </>
// //           )}

// //           <div className="sidebar-sep"/>
// //           <div className="sidebar-foot">
// //             <button className="reindex-btn" onClick={doReindex} disabled={reindexing}>
// //               <Zap size={13} className={reindexing?"spin":""}/>{reindexing?"Indexing…":"Re-index AI"}
// //             </button>
// //             <button className="reindex-btn" style={{marginTop:4,background:"rgba(250,204,21,.07)",borderColor:"rgba(250,204,21,.2)",color:"rgba(250,204,21,.8)"}}
// //               onClick={reprocessEmotions} disabled={reindexing} title="Re-detect emotions on all photos">
// //               <Smile size={13}/> Fix Emotions
// //             </button>
// //             <button className="reindex-btn" style={{marginTop:4,background:"rgba(52,211,153,.07)",borderColor:"rgba(52,211,153,.2)",color:"rgba(52,211,153,.85)"}}
// //               onClick={reprocessColors} disabled={reindexing} title="Recompute dominant colors for color search">
// //               <Palette size={13}/> Fix Colors
// //             </button>
// //             <button className="reindex-btn" style={{marginTop:4,background:"rgba(96,165,250,.07)",borderColor:"rgba(96,165,250,.2)",color:"rgba(96,165,250,.85)"}}
// //               onClick={reprocessNames} disabled={reindexing} title="Auto-detect names from photo captions">
// //               <Users size={13}/> Auto-Name People
// //             </button>
// //             <button className="reindex-btn" style={{marginTop:4,background:"rgba(248,113,113,.07)",borderColor:"rgba(248,113,113,.2)",color:"rgba(248,113,113,.8)"}}
// //               onClick={async()=>{ if(!window.confirm("Delete all empty albums?")) return; const r=await axios.delete(`${API}/albums/empty/cleanup`); setReindexMsg(`✓ Deleted ${r.data.deleted} empty albums`); load(); }}
// //               title="Remove all empty albums">
// //               <Trash2 size={13}/> Clean Empty Albums
// //             </button>
// //             {reindexMsg && <p className="reindex-msg">{reindexMsg}</p>}
// //             <div className="offline-pill"><span className="dot-green"/><span>Fully Offline</span></div>
// //             {voiceErr && voiceErr.includes("setup") && (
// //               <div style={{background:"rgba(245,158,11,.08)",border:"1px solid rgba(245,158,11,.2)",borderRadius:7,padding:"8px 10px",fontSize:10,color:"rgba(245,158,11,.9)",lineHeight:1.5}}>
// //                 <strong>Voice Setup:</strong><br/>
// //                 1. pip install vosk<br/>
// //                 2. Download model from<br/>
// //                 <span style={{color:"#60a5fa",wordBreak:"break-all"}}>alphacephei.com/vosk/models</span><br/>
// //                 3. Extract to:<br/>
// //                 <code style={{fontSize:9,color:"#aaa"}}>models/vosk-model-small-en-us</code>
// //               </div>
// //             )}
// //           </div>
// //         </nav>

// //         {/* ── MAIN ────────────────────────────────────────────────────── */}
// //         <div className="main">

// //           {/* TOPBAR */}
// //           <header className="topbar">
// //             <div className="mode-strip">
// //               {MODES.map(({id,Icon,label})=>(
// //                 <button key={id} className={`mode-btn ${searchMode===id?"mode-btn--on":""}`} onClick={()=>setSearchMode(id)}>
// //                   <Icon size={11}/> {label}
// //                 </button>
// //               ))}
// //               {voice.state!=="idle" && (
// //                 <span className={`voice-chip voice-chip--${voice.state}`}>
// //                   {voice.state==="rec"        && <><MicOff size={10}/> Recording</>}
// //                   {voice.state==="processing" && <><Loader2 size={10} className="spin"/> Transcribing…</>}
// //                   {voice.state==="error"      && <span title={voiceErr}>{voiceErr.length>38?voiceErr.slice(0,38)+"…":voiceErr}</span>}
// //                 </span>
// //               )}
// //             </div>
// //             {/* ── Quick emoji search chips ───────────────────────────────── */}
// //             <div className="emoji-quick-bar">
// //               <span className="emoji-quick-label">Quick:</span>
// //               {[["😊","happy"],["😢","sad"],["😠","angry"],["😲","surprised"],["😨","fearful"],
// //                 ["🐶","dog"],["🏖️","beach"],["🌅","sunset"],["🎉","party"],["✈️","travel"],
// //                 ["❄️","snow"],["🌸","flowers"],["🍕","food"],["🏋️","gym"],["🌊","ocean"]
// //               ].map(([emoji,label])=>(
// //                 <button key={emoji} className="emoji-quick-btn" title={label}
// //                   onClick={()=>{ setSearchQ(emoji); setSearchMode("text"); runSearch("text", emoji); }}>
// //                   {emoji}
// //                 </button>
// //               ))}
// //             </div>
// //             <div className="search-bar">
// //               {(searchMode==="text"||searchMode==="describe"||searchMode==="hybrid") && (
// //                 <div className="search-wrap">
// //                   <Search size={14} className="search-ic"/>
// //                   <input className="search-in" value={searchQ} onChange={e=>setSearchQ(e.target.value)}
// //                     onKeyDown={e=>e.key==="Enter"&&runSearch()}
// //                     placeholder={searchMode==="describe"?"Describe what's in the photo…":searchMode==="hybrid"?"Text + optional image…":"Search by scene, object, person, text…"}
// //                   />
// //                   {voice.ok && (
// //                     <button className={`mic-btn ${voice.state==="rec"?"mic-btn--rec":voice.state==="processing"?"mic-btn--proc":""}`}
// //                       onClick={voice.state==="rec"?voice.stop:voice.start}>
// //                       {voice.state==="rec"?<MicOff size={14}/>:voice.state==="processing"?<Loader2 size={14} className="spin"/>:<Mic size={14}/>}
// //                     </button>
// //                   )}
// //                 </div>
// //               )}
// //               {(searchMode==="image"||searchMode==="hybrid") && (
// //                 <label className="img-search-label">
// //                   {hybridPrev?<img src={hybridPrev} className="img-search-thumb" alt=""/>:<ImgIcon size={13}/>}
// //                   <span>{hybridPrev?"Ready":"Upload ref"}</span>
// //                   <input type="file" accept="image/*" style={{display:"none"}} onChange={e=>{const f=e.target.files?.[0];if(f){setHybridFile(f);setHybridPrev(URL.createObjectURL(f))}}}/>
// //                 </label>
// //               )}
// //               {searchMode==="color" && (
// //                 <div className="color-strip">
// //                   {COLORS.map(c=>(
// //                     <button key={c} title={c} className={`clr-dot ${colorPick===c?"clr-dot--on":""}`}
// //                       style={{background:c==="white"?"#e5e5e5":c==="gray"?"#888":c}}
// //                       onClick={()=>setColorPick(c)}/>
// //                   ))}
// //                 </div>
// //               )}
// //               <button className="btn-go" onClick={()=>runSearch()}>Search</button>
// //               {isGridView && (
// //                 <button className={`btn-batch ${batchMode?"btn-batch--on":""}`} onClick={()=>{setBatchMode(p=>!p);clearSel();}}>
// //                   <CheckSquare size={13}/> {batchMode?"Exit Select":"Select"}
// //                 </button>
// //               )}
// // <button className="btn-upload" onClick={()=>uploadRef.current?.click()}>
// //                 <Upload size={13}/> Upload
// //               </button>
// //               <input ref={uploadRef} type="file" multiple accept="image/*" style={{display:"none"}} onChange={e=>handleUpload(e.target.files)}/>
// //             </div>

// //             {/* Emotion filter bar */}
// //             {isGridView && emoSummary.length>0 && (
// //               <div className="emo-filter-bar">
// //                 <span className="emo-filter-label">Filter:</span>
// //                 {emoSummary.filter(e=>e.emotion&&e.emotion!=="neutral").map(e=>(
// //                   <button key={e.emotion}
// //                     className={`emo-filter-btn ${emoFilter===e.emotion?"emo-filter-btn--on":""}`}
// //                     style={emoFilter===e.emotion?{background:EMO_COLORS[e.emotion]+"22",borderColor:EMO_COLORS[e.emotion]+"66",color:EMO_COLORS[e.emotion]}:{}}
// //                     onClick={()=>filterByEmotion(e.emotion)}>
// //                     {EMO[e.emotion]||"😐"} {e.emotion} <span className="emo-filter-count">{e.count}</span>
// //                   </button>
// //                 ))}
// //                 {(emoFilter||activeTag) && (
// //                   <button className="emo-filter-clear" onClick={()=>{setEmoFilter("");setActiveTag("");load();}}>
// //                     <X size={10}/> Clear
// //                   </button>
// //                 )}
// //               </div>
// //             )}
// //           </header>

// //           {/* BODY */}
// //           <div className="body">
// //             <AnimatePresence mode="wait">
// //               {loading ? (
// //                 <Fade key="load">
// //                   <div className="loader-wrap"><div className="loader-ring"/><p className="loader-text">Loading…</p></div>
// //                 </Fade>

// //               ) : isGridView ? (
// //                 <Fade key={view+emoFilter+activeTag}>
// //                   {/* On This Day banner */}
// //                   {view==="timeline" && onThisDay?.total>0 && (
// //                     <OnThisDayBanner data={onThisDay} expanded={otdExpanded} onToggle={()=>setOtdExpanded(p=>!p)} onOpen={openLb}/>
// //                   )}

// //                   <div className="page-head">
// //                     <h1 className="page-title">
// //                       {view==="search"?`Results for "${searchQ}"`:view==="favorites"?"Favorites":view==="explore"?"Explore":
// //                        activeTag?`#${activeTag}`:emoFilter?`${EMO[emoFilter]} ${emoFilter}`:"Timeline"}
// //                     </h1>
// //                     {images.length>0 && <span className="page-count">{images.length} photos</span>}
// //                     {batchMode && selected.size>0 && <span className="sel-count">{selected.size} selected</span>}
// //                     {view==="explore" && <button className="btn-sm" style={{marginLeft:"auto"}} onClick={load}><Shuffle size={11}/> Shuffle</button>}
// //                   </div>

// //                   {images.length===0
// //                     ? (view==="search" && peopleSuggestions.length>0 ? (
// //                         <div>
// //                           <Empty icon={ImageOff} msg="No photos matched" sub="But these people are in your gallery — click to view their photos"/>
// //                           <div style={{marginTop:16}}>
// //                             <p style={{fontSize:11,color:"var(--mu)",marginBottom:10,fontFamily:"var(--mono)",letterSpacing:".08em"}}>PEOPLE IN YOUR GALLERY:</p>
// //                             <div className="people-grid">
// //                               {peopleSuggestions.map(p=>(
// //                                 <motion.div key={p.id} className="person-card" whileHover={{y:-3}}
// //                                   onClick={async()=>{
// //                                     setLoading(true); setView("search"); setPeopleSuggestions([]);
// //                                     try { const r=await axios.get(`${API}/people/${p.id}`); setImages(r.data.results||r.data.images||[]); }
// //                                     catch {} finally { setLoading(false); }
// //                                   }}>
// //                                   <div className="person-avatar">
// //                                     {p.cover?<img src={imgUrl(p.cover)} alt="" onError={e=>e.target.src=BLANK}/>:<Users size={26} strokeWidth={1} color="#444"/>}
// //                                   </div>
// //                                   <p className="person-name">{p.name}</p>
// //                                   <p className="person-count" style={{color:"var(--ac2)",fontSize:9}}>click to rename</p>
// //                                 </motion.div>
// //                               ))}
// //                             </div>
// //                             <p style={{fontSize:11,color:"#555",marginTop:14,padding:"10px 14px",background:"var(--s2)",borderRadius:8,border:"1px solid var(--br)"}}>
// //                               💡 <strong>Tip:</strong> Click a person above → click ✏ Rename → type their real name → search will work instantly
// //                             </p>
// //                           </div>
// //                         </div>
// //                       ) : <Empty icon={ImageOff} msg={view==="search"?"No results":"No photos yet"} sub={view!=="search"?"Upload photos to get started":"Try different keywords"} onClick={view!=="search"?()=>uploadRef.current?.click():null}/>)
// //                     : <ImgGrid list={images} onOpen={i=>openLb(images,i)} onFav={toggleFav} onDel={softDelete} batchMode={batchMode} selected={selected} onToggle={toggleSelect}/>
// //                   }
// //                 </Fade>

// //               ) : view==="faces" ? (
// //                 <Fade key="faces">
// //                   <PageHead title="People" count={faces.length}/>
// //                   {faces.length===0
// //                     ? <Empty icon={Users} msg="No people detected" sub="Run Re-index to detect faces"/>
// //                     : <div className="people-grid">{faces.map(p=><PersonCard key={p.id} p={p} onClick={()=>openPerson(p)}/>)}</div>
// //                   }
// //                 </Fade>

// //               ) : view==="albums" ? (
// //                 <Fade key="albums">
// //                   <div className="page-head">
// //                     <h1 className="page-title">Albums</h1>
// //                     {albums.length>0 && <span className="page-count">{albums.length}</span>}
// //                     <button className="btn-sm btn-sm--accent" style={{marginLeft:"auto"}} onClick={()=>{ setPendingAlbumImages([]); setCreateAlbumOpen(true); }}>
// //                       <Plus size={12}/> New Album
// //                     </button>
// //                     <button className="btn-sm" style={{marginLeft:6,background:"var(--s3)",border:"1px solid var(--br)"}} onClick={()=>setCreateEventOpen(true)}>🎉 New Event</button>
// //                   </div>
// //                   {albums.length===0
// //                     ? <Empty icon={BookImage} msg="No albums" sub="Run Re-index to auto-generate or create manually"/>
// //                     : <div className="albums-grid">{albums.map(a=><AlbumCard key={a.id} a={a} onClick={()=>openAlbum(a)} onDelete={a.type==="manual"?()=>deleteAlbum(a.id):null}/>)}</div>
// //                   }
// //                 </Fade>

// //               ) : view==="duplicates" ? (
// //                 <Fade key="dupes">
// //                   <PageHead title="Duplicate Groups" count={dupes.length} unit="groups"/>
// //                   {dupes.length===0
// //                     ? <Empty icon={Copy} msg="No duplicates" sub="All images are unique"/>
// //                     : <div className="dupe-list">
// //                         {dupes.map((g,gi)=>(
// //                           <div key={gi} className="dupe-group">
// //                             <div className="dupe-head">
// //                               <span className="tag-sm">Group {gi+1}</span>
// //                               <span className="tag-sm tag-sm--accent">{g.count} duplicates</span>
// //                               {g.total_size&&<span className="tag-sm">{(g.total_size/1024/1024).toFixed(1)} MB</span>}
// //                             </div>
// //                             <ImgGrid list={g.images||[]} compact onOpen={i=>openLb(g.images,i)} onFav={toggleFav} onDel={softDelete}/>
// //                           </div>
// //                         ))}
// //                       </div>
// //                   }
// //                 </Fade>

// //               ) : view==="trash" ? (
// //                 <Fade key="trash">
// //                   <PageHead title="Trash" count={trash.length}/>
// //                   {trash.length===0 ? <Empty icon={Trash} msg="Trash is empty"/> : (
// //                     <div className="img-grid">
// //                       {trash.map(img=>(
// //                         <div key={img.id} className="trash-card">
// //                           <img src={imgUrl(img.filename)} alt="" className="trash-img" onError={e=>e.target.src=BLANK}/>
// //                           <div className="trash-overlay">
// //                             <button className="trash-btn trash-btn--restore" onClick={()=>restoreImg(img.id)}><RotateCcw size={11}/> Restore</button>
// //                             <button className="trash-btn trash-btn--del" onClick={()=>permDelete(img.id)}><AlertTriangle size={11}/> Delete Forever</button>
// //                           </div>
// //                           {img.trashed_at&&<span className="trash-date">{new Date(img.trashed_at).toLocaleDateString()}</span>}
// //                         </div>
// //                       ))}
// //                     </div>
// //                   )}
// //                 </Fade>

// //               ) : view==="stats" ? (
// //                 <Fade key="stats"><StatsPage stats={stats} peopleFreq={peopleFreq}/></Fade>
// //               ) : null}
// //             </AnimatePresence>
// //           </div>

// //           {/* BATCH ACTION BAR */}
// //           <AnimatePresence>
// //             {batchMode && (
// //               <motion.div className="batch-bar"
// //                 initial={{y:60,opacity:0}} animate={{y:0,opacity:1}} exit={{y:60,opacity:0}}>
// //                 <div className="batch-bar-left">
// //                   <span className="batch-count"><CheckCheck size={14}/> {selected.size} selected</span>
// //                   <button className="batch-sm" onClick={selectAll}>All</button>
// //                   <button className="batch-sm" onClick={clearSel}>None</button>
// //                 </div>
// //                 <div className="batch-bar-actions">
// //                   <button className="batch-action-btn batch-action-btn--fav" onClick={()=>batchFavorite(true)} title="Favorite all"><Heart size={13}/> Favorite</button>
// //                   <button className="batch-action-btn batch-action-btn--del" onClick={batchDelete} title="Trash all"><Trash2 size={13}/> Trash</button>
// //                   <div className="batch-input-group">
// //                     <input className="batch-input" placeholder="Add tag…" value={batchTagInput} onChange={e=>setBatchTagInput(e.target.value)} onKeyDown={e=>e.key==="Enter"&&batchTag()}/>
// //                     <button className="batch-action-btn" onClick={batchTag}><Tag size={12}/></button>
// //                   </div>
// //                   <div className="batch-input-group">
// //                     <select className="batch-select" value={batchAlbumId} onChange={e=>setBatchAlbumId(e.target.value)}>
// //                       <option value="">Add to album…</option>
// //                       {albums.map(a=><option key={a.id} value={a.id}>{a.title}</option>)}
// //                     </select>
// //                     <button className="batch-action-btn" onClick={batchAddAlbum} title="Add to existing album"><Layers size={12}/></button>
// //                   </div>
// //                   <button className="batch-action-btn batch-action-btn--album"
// //                     onClick={()=>{
// //                       setPendingAlbumImages([...selected]); // snapshot NOW before modal opens
// //                       setCreateAlbumOpen(true);
// //                     }}
// //                     title={`Create new album with ${selected.size} selected photos`}
// //                     disabled={selected.size===0}>
// //                     <FolderPlus size={12}/> New Album
// //                   </button>
// //                 </div>
// //               </motion.div>
// //             )}
// //           </AnimatePresence>
// //         </div>
// //       </div>

// //       {/* LIGHTBOX */}
// //       <AnimatePresence>
// //         {lb && <Lightbox list={lb.list} idx={lb.idx} onClose={closeLb} onShift={shiftLb} onFav={toggleFav} onDel={(id,e)=>{softDelete(id,e);closeLb();}} onAddTag={addTagToImage} onRemoveTag={removeTagFromImage} albums={albums}/>}
// //       </AnimatePresence>

// //       {/* PERSON MODAL */}
// //       <AnimatePresence>
// //         {personModal && <PersonModal data={personModal} onClose={()=>setPersonModal(null)} onRename={renamePerson} onOpen={(list,i)=>{setPersonModal(null);setTimeout(()=>openLb(list,i),80);}}/>}
// //       </AnimatePresence>

// //       {/* ALBUM MODAL */}
// //       <AnimatePresence>
// //         {albumModal && <AlbumModal data={albumModal} onClose={()=>setAlbumModal(null)} onRename={renameAlbum} onDelete={deleteAlbum} onOpen={(list,i)=>{setAlbumModal(null);setTimeout(()=>openLb(list,i),80);}}/>}
// //       </AnimatePresence>

// //       {/* CREATE ALBUM MODAL */}
// //       <AnimatePresence>
// //         {createAlbumOpen && (
// //           <ModalWrap onClose={()=>setCreateAlbumOpen(false)}>
// //             <div className="modal-head">
// //               <h2 className="modal-title">New Album</h2>
// //               <button className="lb-hbtn lb-close" onClick={()=>setCreateAlbumOpen(false)}><X size={17}/></button>
// //             </div>
// //             {pendingAlbumImages.length>0 && (
// //               <div style={{background:"rgba(96,165,250,.08)",border:"1px solid rgba(96,165,250,.2)",borderRadius:8,padding:"8px 12px",marginBottom:12,fontSize:12,color:"rgba(96,165,250,.9)"}}>
// //                 📸 <strong>{pendingAlbumImages.length} selected photos</strong> will be added to this album
// //               </div>
// //             )}
// //             <div className="create-album-form">
// //               <input className="form-input" placeholder="Album title…" value={newAlbumTitle}
// //                 onChange={e=>setNewAlbumTitle(e.target.value)}
// //                 onKeyDown={e=>e.key==="Enter"&&newAlbumTitle.trim()&&createAlbum()}
// //                 autoFocus/>
// //               <textarea className="form-textarea" placeholder="Description (optional)…"
// //                 value={newAlbumDesc} onChange={e=>setNewAlbumDesc(e.target.value)} rows={2}/>
// //               <div style={{display:"flex",gap:8}}>
// //                 <button className="btn-sm btn-sm--primary" style={{flex:1}} onClick={()=>createAlbum()} disabled={!newAlbumTitle.trim()}>
// //                   <FolderPlus size={13}/> Create Album
// //                 </button>
// //                 <button className="btn-sm" onClick={()=>setCreateAlbumOpen(false)}>Cancel</button>
// //               </div>
// //             </div>
// //           </ModalWrap>
// //         )}
// //       </AnimatePresence>

// //       {/* CREATE NAMED EVENT MODAL */}
// //       <AnimatePresence>
// //         {createEventOpen && (
// //           <ModalWrap onClose={()=>setCreateEventOpen(false)}>
// //             <div className="modal-head">
// //               <h2 className="modal-title">🎉 New Event</h2>
// //               <button className="lb-hbtn lb-close" onClick={()=>setCreateEventOpen(false)}><X size={17}/></button>
// //             </div>
// //             <div className="create-album-form">
// //               <label style={{fontSize:11,color:"var(--mu)",marginBottom:4}}>EVENT TYPE</label>
// //               <div style={{display:"flex",flexWrap:"wrap",gap:6,marginBottom:12}}>
// //                 {["Birthday","Trip","Vacation","Wedding","Anniversary","Graduation","Party","Holiday","Family","Work","Other"].map(t=>(
// //                   <button key={t}
// //                     className={"btn-sm" + (newEventType===t?" btn-sm--primary":"")}
// //                     style={{fontSize:11,padding:"4px 10px"}}
// //                     onClick={()=>setNewEventType(t)}>
// //                     {t==="Birthday"?"🎂":t==="Trip"?"✈️":t==="Vacation"?"🏖️":t==="Wedding"?"💍":t==="Anniversary"?"💕":t==="Graduation"?"🎓":t==="Party"?"🎊":t==="Holiday"?"🎄":t==="Family"?"👨‍👩‍👧":t==="Work"?"💼":"📅"} {t}
// //                   </button>
// //                 ))}
// //               </div>
// //               <input className="form-input" placeholder="Event name (e.g. Mom's 60th Birthday)…" value={newEventTitle} onChange={e=>setNewEventTitle(e.target.value)} autoFocus/>
// //               <input className="form-input" type="date" value={newEventDate} onChange={e=>setNewEventDate(e.target.value)} style={{marginTop:8}}/>
// //               <textarea className="form-textarea" placeholder="Notes (optional)…" value={newEventDesc} onChange={e=>setNewEventDesc(e.target.value)} rows={2} style={{marginTop:8}}/>
// //               <button className="btn-sm btn-sm--primary" onClick={createNamedEvent} disabled={!newEventTitle.trim()} style={{marginTop:4}}>
// //                 🎉 Create Event
// //               </button>
// //             </div>
// //           </ModalWrap>
// //         )}
// //       </AnimatePresence>

// //       {/* CO-OCCURRENCE MODAL */}
// //       <AnimatePresence>
// //         {showCoModal && (
// //           <ModalWrap onClose={()=>setShowCoModal(false)}>
// //             <div className="modal-head">
// //               <h2 className="modal-title"><GitMerge size={16} style={{marginRight:6,verticalAlign:"middle"}}/>Co-Appearance Search</h2>
// //               <button className="lb-hbtn lb-close" onClick={()=>setShowCoModal(false)}><X size={17}/></button>
// //             </div>
// //             <CoOccurrencePanel faces={faces} onSearch={runCoOccurrence}/>
// //           </ModalWrap>
// //         )}
// //       </AnimatePresence>

// //       {/* FACE SIMILARITY MODAL */}
// //       <AnimatePresence>
// //         {showFaceSim && (
// //           <ModalWrap onClose={()=>setShowFaceSim(false)}>
// //             <div className="modal-head">
// //               <h2 className="modal-title"><UserCheck size={16} style={{marginRight:6,verticalAlign:"middle"}}/>Face Similarity Search</h2>
// //               <button className="lb-hbtn lb-close" onClick={()=>setShowFaceSim(false)}><X size={17}/></button>
// //             </div>
// //             <div className="create-album-form">
// //               <p style={{fontSize:12,color:"var(--mu)",marginBottom:12}}>Upload a face crop or photo — find all photos with the same person.</p>
// //               {faceSrcPrev && <img src={faceSrcPrev} style={{width:120,height:120,objectFit:"cover",borderRadius:8,marginBottom:12,border:"2px solid var(--ac)"}} alt="face"/>}
// //               <input type="file" accept="image/*" style={{display:"none"}} id="face-sim-input"
// //                 onChange={e=>{
// //                   const f=e.target.files[0]; if(!f) return;
// //                   setFaceSrcFile(f);
// //                   const r=new FileReader(); r.onload=ev=>setFaceSrcPrev(ev.target.result); r.readAsDataURL(f);
// //                 }}/>
// //               <button className="btn-sm" onClick={()=>document.getElementById("face-sim-input").click()}>
// //                 <Camera size={12}/> {faceSrcPrev?"Change Photo":"Upload Face Photo"}
// //               </button>
// //               {faceSrcFile && (
// //                 <button className="btn-sm btn-sm--primary" style={{marginTop:10}} onClick={runFaceSimilarity}>
// //                   <UserCheck size={12}/> Find Similar Faces
// //                 </button>
// //               )}
// //             </div>
// //           </ModalWrap>
// //         )}
// //       </AnimatePresence>
// //     </>
// //   );
// // }

// // /* ══════════════════════════════════════════════════════════════════════════ */
// // /* ON THIS DAY BANNER                                                          */
// // /* ══════════════════════════════════════════════════════════════════════════ */
// // function OnThisDayBanner({ data, expanded, onToggle, onOpen }) {
// //   return (
// //     <div className="otd-banner">
// //       <div className="otd-header" onClick={onToggle}>
// //         <div className="otd-left">
// //           <Calendar size={15} color="#f59e0b"/>
// //           <div>
// //             <span className="otd-title">On This Day — {data.date}</span>
// //             <span className="otd-sub">{data.total} photo{data.total!==1?"s":""} from {data.years?.length} year{data.years?.length!==1?"s":""} ago</span>
// //           </div>
// //         </div>
// //         <button className="otd-toggle">{expanded?<ChevronUp size={14}/>:<ChevronDown size={14}/>}</button>
// //       </div>
// //       <AnimatePresence>
// //         {expanded && (
// //           <motion.div initial={{height:0,opacity:0}} animate={{height:"auto",opacity:1}} exit={{height:0,opacity:0}}>
// //             {data.years?.map(yr=>(
// //               <div key={yr.year} className="otd-year">
// //                 <p className="otd-year-label">{yr.year} — {yr.count} photo{yr.count!==1?"s":""}</p>
// //                 <div className="otd-strip">
// //                   {yr.images.map((img,i)=>(
// //                     <div key={img.id} className="otd-thumb" onClick={()=>onOpen(yr.images,i)}>
// //                       <img src={imgUrl(img.filename)} alt="" onError={e=>e.target.src=BLANK}/>
// //                       {img.caption_short && <div className="otd-thumb-caption">{img.caption_short}</div>}
// //                     </div>
// //                   ))}
// //                 </div>
// //               </div>
// //             ))}
// //           </motion.div>
// //         )}
// //       </AnimatePresence>
// //     </div>
// //   );
// // }


// // /* ══════════════════════════════════════════════════════════════════════════ */
// // /* IMAGE GRID                                                                  */
// // /* ══════════════════════════════════════════════════════════════════════════ */
// // function ImgGrid({ list, onOpen, onFav, onDel, compact, batchMode, selected, onToggle }) {
// //   return (
// //     <div className={`img-grid${compact?" img-grid--sm":""}`}>
// //       {list.map((img,i)=>(
// //         <PhotoCard key={img.id??i} img={img} onOpen={()=>onOpen(i)} onFav={onFav} onDel={onDel}
// //           batchMode={batchMode} isSelected={selected?.has(img.id)} onToggle={onToggle}/>
// //       ))}
// //     </div>
// //   );
// // }

// // function PhotoCard({ img, onOpen, onFav, onDel, batchMode, isSelected, onToggle }) {
// //   const q=qInfo(img.quality_level); const emo=EMO[img.dominant_emotion]||"";
// //   return (
// //     <motion.div className={`photo-card ${isSelected?"photo-card--sel":""}`}
// //       whileHover={{y:-2}} transition={{duration:.15}}
// //       onClick={batchMode?()=>onToggle(img.id):onOpen}>
// //       <img src={imgUrl(img.filename)||BLANK} alt="" className="photo-img" loading="lazy" onError={e=>{e.target.onerror=null;e.target.src=BLANK;}}/>
// //       {img.quality_level&&img.quality_level!=="Processing..."&&(
// //         <span className="chip chip--q" style={{color:q.c,background:q.bg,borderColor:q.c+"33"}}>{img.quality_level[0]}</span>
// //       )}
// //       {emo&&img.dominant_emotion!=="neutral"&&<span className="chip chip--emo">{emo}</span>}
// //       {img.is_favorite&&<span className="fav-dot">♥</span>}
// //       {batchMode&&(
// //         <div className="batch-check">{isSelected?<CheckSquare size={18} color="#7c6af7"/>:<Square size={18} color="rgba(255,255,255,.5)"/>}</div>
// //       )}
// //       {!batchMode&&(
// //         <div className="photo-hover">
// //           <div className="photo-actions">
// //             <button className="ph-btn ph-btn--fav" onClick={e=>{e.stopPropagation();onFav(img.id,e);}}><Heart size={11}/></button>
// //             <button className="ph-btn ph-btn--del" onClick={e=>{e.stopPropagation();onDel(img.id,e);}}><Trash2 size={11}/></button>
// //           </div>
// //           <div className="photo-info">
// //             {img.caption_short&&<p className="photo-caption">{img.caption_short}</p>}
// //             <div className="photo-meta-row">
// //               {img.quality_level&&img.quality_level!=="Processing..."&&<span className="meta-chip" style={{color:q.c}}>◆ {img.quality_level}</span>}
// //               {img.aesthetic_score>0&&<span className="meta-chip" style={{color:"#fbbf24"}}>★ {Number(img.aesthetic_score).toFixed(1)}</span>}
// //               {img.dominant_emotion&&img.dominant_emotion!=="neutral"&&<span className="meta-chip">{emo} {img.dominant_emotion}</span>}
// //               {img.user_tags?.length>0&&<span className="meta-chip" style={{color:"#86efac"}}>#{img.user_tags[0]}{img.user_tags.length>1?` +${img.user_tags.length-1}`:""}</span>}
// //             </div>
// //             <div className="view-hint"><Eye size={9}/> Open</div>
// //           </div>
// //         </div>
// //       )}
// //     </motion.div>
// //   );
// // }

// // /* ══════════════════════════════════════════════════════════════════════════ */
// // /* LIGHTBOX                                                                    */
// // /* ══════════════════════════════════════════════════════════════════════════ */
// // function Lightbox({ list, idx, onClose, onShift, onFav, onDel, onAddTag, onRemoveTag, albums }) {
// //   const img=list[idx]; if(!img) return null;
// //   const q=qInfo(img.quality_level); const emo=EMO[img.dominant_emotion]||"😐";
// //   const [tagInput, setTagInput] = useState("");
// //   const [imgTags,  setImgTags]  = useState(img.user_tags||[]);
// //   const [addAlbum, setAddAlbum] = useState("");
// //   const [note,     setNote]     = useState(img.photo_note||"");
// //   const [noteEditing, setNoteEditing] = useState(false);
// //   const [noteSaving,  setNoteSaving]  = useState(false);

// //   // Always fetch latest tags + note from server when photo changes
// //   useEffect(()=>{
// //     setTagInput(""); setNoteEditing(false);
// //     setImgTags(img.user_tags||[]);
// //     setNote(img.photo_note||"");
// //     axios.get(`${API}/photo/${img.id}/tags`)
// //       .then(r => setImgTags(r.data.tags||[]))
// //       .catch(()=>{});
// //     axios.get(`${API}/photo/${img.id}/note`)
// //       .then(r => setNote(r.data.note||""))
// //       .catch(()=>{});
// //   }, [img.id]);

// //   const saveNote = async () => {
// //     setNoteSaving(true);
// //     const fd=new FormData(); fd.append("note", note);
// //     try { await axios.post(`${API}/photo/${img.id}/note`, fd); setNoteEditing(false); }
// //     catch(e) { alert("Failed to save note"); }
// //     setNoteSaving(false);
// //   };

// //   const doAddTag = async () => {
// //     if (!tagInput.trim()) return;
// //     const newTags = await onAddTag(img.id, tagInput.trim());
// //     if (newTags) setImgTags(newTags);
// //     else setImgTags(p=>[...new Set([...p, tagInput.trim().toLowerCase()])]);
// //     setTagInput("");
// //   };
// //   const doRemoveTag = async tag => {
// //     const newTags = await onRemoveTag(img.id, tag);
// //     if (newTags) setImgTags(newTags);
// //     else setImgTags(p=>p.filter(t=>t!==tag));
// //   };
// //   const doAddToAlbum = async () => {
// //     if (!addAlbum) return;
// //     const fd=new FormData(); fd.append("image_ids",img.id);
// //     try { await axios.post(`${API}/batch/album`,fd.set?fd:(()=>{fd.append("album_id",addAlbum);return fd;})()); } catch {}
// //     setAddAlbum("");
// //   };

// //   return (
// //     <motion.div className="lb-bg" initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}} onClick={onClose}>
// //       <motion.div className="lb-box" initial={{scale:.96}} animate={{scale:1}} exit={{scale:.96}} onClick={e=>e.stopPropagation()}>
// //         <div className="lb-left">
// //           {idx>0&&<button className="lb-nav lb-nav--l" onClick={()=>onShift(-1)}><ChevronLeft size={20}/></button>}
// //           <AnimatePresence mode="wait">
// //             <motion.img key={img.id??idx} src={imgUrl(img.filename)||BLANK} alt="" className="lb-img"
// //               initial={{opacity:0,x:16}} animate={{opacity:1,x:0}} exit={{opacity:0,x:-16}} transition={{duration:.18}}
// //               onError={e=>{e.target.onerror=null;e.target.src=BLANK;}}/>
// //           </AnimatePresence>
// //           {idx<list.length-1&&<button className="lb-nav lb-nav--r" onClick={()=>onShift(1)}><ChevronRight size={20}/></button>}
// //           <div className="lb-counter">{idx+1} / {list.length}</div>
// //           <div className="lb-hint">← → · ESC</div>
// //         </div>

// //         <div className="lb-right">
// //           <div className="lb-right-head">
// //             <h3 className="lb-right-title">Details</h3>
// //             <div className="lb-head-btns">
// //               <button className="lb-hbtn" onClick={()=>onFav(img.id)} style={{color:img.is_favorite?"#f43f5e":undefined}}>
// //                 <Heart size={15} fill={img.is_favorite?"#f43f5e":"none"} stroke={img.is_favorite?"#f43f5e":"currentColor"}/>
// //               </button>
// //               <button className="lb-hbtn" onClick={e=>onDel(img.id,e)}><Trash2 size={15}/></button>
// //               <button className="lb-hbtn lb-close" onClick={onClose}><X size={17}/></button>
// //             </div>
// //           </div>

// //           <div className="lb-scroll">
// //             {img.caption_short&&(
// //               <DB icon={<Camera size={12}/>} label="AI Caption" c="#818cf8">
// //                 <p className="db-caption">{img.caption_short}</p>
// //                 {img.caption_detailed&&img.caption_detailed!==img.caption_short&&<p className="db-caption-sub">{img.caption_detailed}</p>}
// //               </DB>
// //             )}
// //             {img.quality_score>0&&(
// //               <DB icon={<TrendingUp size={12}/>} label="Quality" c={q.c}>
// //                 <div className="q-wrap">
// //                   <div className="q-circle" style={{borderColor:q.c}}>
// //                     <span style={{color:q.c,fontSize:18,fontWeight:800}}>{Math.round(img.quality_score)}</span>
// //                   </div>
// //                   <div className="q-right">
// //                     <p className="q-level" style={{color:q.c}}>{img.quality_level}</p>
// //                     {img.sharpness>0&&<><QBar label="Sharpness" val={img.sharpness} c={q.c}/><QBar label="Exposure" val={img.exposure} c={q.c}/><QBar label="Contrast" val={img.contrast} c={q.c}/><QBar label="Composition" val={img.composition} c={q.c}/></>}
// //                   </div>
// //                 </div>
// //               </DB>
// //             )}
// //             {img.aesthetic_score>0&&(
// //               <DB icon={<Star size={12}/>} label="Aesthetic" c="#fbbf24">
// //                 <div className="aes-row"><span className="aes-score">★ {Number(img.aesthetic_score).toFixed(1)}</span>{img.aesthetic_rating&&<span className="aes-rating">{img.aesthetic_rating}</span>}</div>
// //               </DB>
// //             )}
// //             {img.dominant_emotion&&img.dominant_emotion!=="neutral"&&(
// //               <DB icon={<Smile size={12}/>} label="Emotion" c={EMO_COLORS[img.dominant_emotion]||"#f9a8d4"}>
// //                 <div className="emo-row">
// //                   <span style={{fontSize:30}}>{EMO[img.dominant_emotion]}</span>
// //                   <div><p className="emo-name">{img.dominant_emotion}</p>{img.face_emotion_count>0&&<p className="emo-sub">{img.face_emotion_count} face{img.face_emotion_count>1?"s":""}</p>}</div>
// //                 </div>
// //               </DB>
// //             )}
// //             {img.ocr_text_enhanced?.trim()&&(
// //               <DB icon={<Type size={12}/>} label="Text in Photo" c="#22d3ee">
// //                 <p className="ocr-text">{img.ocr_text_enhanced}</p>
// //               </DB>
// //             )}

// //             {/* Tags block */}
// //             <DB icon={<FileText size={12}/>} label="Personal Note" c="#fcd34d">
// //               {noteEditing ? (
// //                 <div>
// //                   <textarea className="form-textarea" value={note} onChange={e=>setNote(e.target.value)}
// //                     placeholder="Write a personal note about this photo…" rows={3} autoFocus
// //                     style={{marginBottom:6,fontSize:12}}/>
// //                   <div style={{display:"flex",gap:6}}>
// //                     <button className="btn-sm btn-sm--primary" onClick={saveNote} disabled={noteSaving}>
// //                       {noteSaving?"Saving…":"Save Note"}
// //                     </button>
// //                     <button className="btn-sm" onClick={()=>setNoteEditing(false)}>Cancel</button>
// //                   </div>
// //                 </div>
// //               ) : (
// //                 <div>
// //                   {note ? <p style={{fontSize:12,color:"var(--tx)",lineHeight:1.5,marginBottom:6}}>{note}</p>
// //                         : <p style={{fontSize:11,color:"var(--mu)"}}>No note yet</p>}
// //                   <button className="btn-sm" style={{fontSize:10,padding:"3px 8px"}} onClick={()=>setNoteEditing(true)}>
// //                     ✏ {note?"Edit Note":"Add Note"}
// //                   </button>
// //                 </div>
// //               )}
// //             </DB>

// //             <DB icon={<Hash size={12}/>} label="Tags" c="#86efac">
// //               <div className="tag-chips">
// //                 {imgTags.map(t=>(
// //                   <span key={t} className="tag-chip-item">
// //                     #{t}
// //                     <button className="tag-chip-x" onClick={()=>doRemoveTag(t)}><X size={8}/></button>
// //                   </span>
// //                 ))}
// //                 {imgTags.length===0&&<span style={{fontSize:10,color:"var(--mu)"}}>No tags yet</span>}
// //               </div>
// //               <div className="tag-add-row">
// //                 <input className="tag-add-input" placeholder="Add tag…" value={tagInput} onChange={e=>setTagInput(e.target.value)} onKeyDown={e=>e.key==="Enter"&&doAddTag()}/>
// //                 <button className="tag-add-btn" onClick={doAddTag}><Plus size={12}/></button>
// //               </div>
// //             </DB>

// //             {/* Add to album */}
// //             {albums?.length>0&&(
// //               <DB icon={<Layers size={12}/>} label="Add to Album" c="#a78bfa">
// //                 <div className="tag-add-row">
// //                   <select className="batch-select" style={{flex:1}} value={addAlbum} onChange={e=>setAddAlbum(e.target.value)}>
// //                     <option value="">Choose album…</option>
// //                     {albums.map(a=><option key={a.id} value={a.id}>{a.title}</option>)}
// //                   </select>
// //                   <button className="tag-add-btn" onClick={doAddToAlbum}><Plus size={12}/></button>
// //                 </div>
// //               </DB>
// //             )}

// //             {img.scene_label&&(
// //               <DB icon={<Tag size={12}/>} label="Objects" c="#86efac">
// //                 <div className="tag-wrap">{img.scene_label.split(",").map((t,i)=><span key={i} className="obj-tag">{t.trim()}</span>)}</div>
// //               </DB>
// //             )}
// //             <DB icon={<Info size={12}/>} label="Metadata" c="#6b7280">
// //               <div className="meta-stack">
// //                 {img.timestamp&&<MRow k="Captured" v={new Date(img.timestamp).toLocaleString()}/>}
// //                 {img.width&&<MRow k="Dimensions" v={`${img.width} × ${img.height}`}/>}
// //                 {img.person_count>0&&<MRow k="People" v={`${img.person_count} detected`}/>}
// //                 {img.score>0&&<MRow k="Relevance" v={`${Math.round(img.score)}%`}/>}
// //               </div>
// //             </DB>
// //           </div>
// //         </div>
// //       </motion.div>
// //     </motion.div>
// //   );
// // }

// // function QBar({label,val,c}){ return <div className="qbar-row"><span className="qbar-label">{label}</span><div className="qbar-track"><div className="qbar-fill" style={{width:`${Math.min(100,(val||0)*100)}%`,background:c}}/></div><span className="qbar-num">{Math.round((val||0)*100)}</span></div>; }
// // function MRow({k,v}){ return <div className="mrow"><span className="mrow-k">{k}</span><span className="mrow-v">{v}</span></div>; }
// // function DB({icon,label,c,children}){ return <div className="db" style={{"--dc":c}}><div className="db-head">{React.cloneElement(icon,{color:c})}<span className="db-label">{label}</span></div><div className="db-body">{children}</div></div>; }

// // /* Person / Album components */
// // function PersonCard({p,onClick}){ return <motion.div className="person-card" whileHover={{y:-3}} onClick={onClick}><div className="person-avatar">{p.cover?<img src={imgUrl(p.cover)} alt="" onError={e=>e.target.src=BLANK}/>:<Users size={26} strokeWidth={1} color="#444"/>}</div><p className="person-name">{p.name}</p><p className="person-count">{p.count||p.face_count||0} photos</p></motion.div>; }

// // function PersonModal({data,onClose,onRename,onOpen}){
// //   const [editing, setEditing] = useState(false);
// //   const [name, setName] = useState(data.name || "");
// //   const imgs = data.images || [];
// //   const isDefault = /^Person \d+$/.test(data.name || "");

// //   const doSave = () => {
// //     if (name.trim()) { onRename(data.id, name.trim()); setEditing(false); }
// //   };

// //   return (
// //     <ModalWrap onClose={onClose} wide>
// //       <div className="modal-head">
// //         <div>
// //           <h2 className="modal-title">{data.name}</h2>
// //           <p className="modal-sub">{data.face_count||data.count||0} photos</p>
// //         </div>
// //         <div style={{display:"flex",gap:6,alignItems:"center"}}>
// //           <button className="btn-sm btn-sm--primary" onClick={()=>{setEditing(true);setName(isDefault?"":data.name);}}>
// //             ✏ Rename
// //           </button>
// //           <button className="lb-hbtn lb-close" onClick={onClose}><X size={17}/></button>
// //         </div>
// //       </div>

// //       {isDefault && !editing && (
// //         <div style={{background:"rgba(250,204,21,.07)",border:"1px solid rgba(250,204,21,.2)",borderRadius:8,padding:"10px 14px",marginBottom:14}}>
// //           <p style={{fontSize:12,color:"rgba(250,204,21,.9)"}}>
// //             💡 <strong>Give this person a name</strong> so you can search for their photos. Click <strong>✏ Rename</strong> above.
// //           </p>
// //         </div>
// //       )}

// //       {editing && (
// //         <div style={{background:"var(--s2)",border:"1px solid var(--br)",borderRadius:10,padding:"14px 16px",marginBottom:14}}>
// //           <p style={{fontSize:11,color:"var(--mu)",marginBottom:8,fontFamily:"var(--mono)",letterSpacing:".07em"}}>
// //             ENTER PERSON'S NAME
// //           </p>
// //           <div style={{display:"flex",gap:8,alignItems:"center"}}>
// //             <input
// //               className="rename-in"
// //               style={{flex:1,fontSize:15,padding:"10px 14px"}}
// //               value={name}
// //               onChange={e=>setName(e.target.value)}
// //               onKeyDown={e=>{if(e.key==="Enter")doSave(); if(e.key==="Escape")setEditing(false);}}
// //               placeholder="e.g. Vijay, Shah Rukh Khan…"
// //               autoFocus
// //             />
// //             <button className="btn-sm btn-sm--primary" style={{padding:"10px 18px",fontSize:13}} onClick={doSave} disabled={!name.trim()}>
// //               Save
// //             </button>
// //             <button className="btn-sm" onClick={()=>setEditing(false)}>Cancel</button>
// //           </div>
// //           <p style={{fontSize:10,color:"#555",marginTop:8}}>
// //             After saving, you can search for this name in the search bar to find all their photos.
// //           </p>
// //         </div>
// //       )}

// //       {data.images===null
// //         ? <div className="loader-wrap" style={{minHeight:140}}><div className="loader-ring"/></div>
// //         : imgs.length>0
// //           ? <ImgGrid list={imgs} onOpen={i=>onOpen(imgs,i)} onFav={()=>{}} onDel={()=>{}}/>
// //           : <p className="empty-sub">No photos for this person</p>
// //       }
// //     </ModalWrap>
// //   );
// // }

// // function AlbumCard({a,onClick,onDelete}){
// //   return (
// //     <motion.div className="album-card" whileHover={{y:-3,scale:1.01}} onClick={onClick}>
// //       {a.cover
// //         ? <img src={imgUrl(a.cover)} alt="" className="album-cover" onError={e=>e.target.src=BLANK}/>
// //         : <div className="album-cover-ph"><FolderPlus size={32} strokeWidth={1} color="#555"/><span style={{fontSize:10,color:"#555",marginTop:4}}>Empty</span></div>
// //       }
// //       {a.thumbnails?.length>1&&<div className="album-strip">{a.thumbnails.slice(1,5).map((fn,i)=><img key={i} src={imgUrl(fn)} alt="" className="album-strip-img" onError={e=>e.target.style.display="none"}/>)}</div>}
// //       <div className="album-info">
// //         <h3 className="album-title">{a.title}</h3>
// //         <div className="album-meta">
// //           {(()=>{
// //   const typeEmoji = {Birthday:"🎂",Trip:"✈️",Vacation:"🏖️",Wedding:"💍",Anniversary:"💕",Graduation:"🎓",Party:"🎊",Holiday:"🎄",Family:"👨‍👩‍👧",Work:"💼"};
// //   const desc = a.description||"";
// //   const etype = Object.keys(typeEmoji).find(k=>desc.startsWith(k));
// //   const label = a.type==="manual"?"✎ album": etype ? typeEmoji[etype]+" "+etype : a.type||"event";
// //   const cls = a.type==="manual"?"tag-sm--manual":"tag-sm--accent";
// //   return <span className={`tag-sm ${cls}`}>{label}</span>;
// // })()}
// //           {a.date&&<span className="tag-sm">{a.date}</span>}
// //           <span className="tag-sm">📸 {a.count}</span>
// //         </div>
// //       </div>
// //       {onDelete&&<button className="album-del-btn" onClick={e=>{e.stopPropagation();onDelete();}} title="Delete album"><X size={12}/></button>}
// //     </motion.div>
// //   );
// // }

// // function AlbumModal({data,onClose,onOpen,onRename,onDelete}){
// //   const imgs=data.images||[];
// //   const [editing,setEditing]=React.useState(false);
// //   const [title,setTitle]=React.useState(data.title||"");
// //   const [desc,setDesc]=React.useState(data.description||"");
// //   const saveRename=()=>{ onRename&&onRename(data.id,title,desc); setEditing(false); };
// //   return (
// //     <ModalWrap onClose={onClose} wide>
// //       <div className="modal-head">
// //         <div>
// //           <h2 className="modal-title">{data.title}</h2>
// //           <p className="modal-sub">{data.image_count||data.count||0} photos{data.date?` · ${data.date}`:""}</p>
// //         </div>
// //         <div style={{display:"flex",gap:6,alignItems:"center"}}>
// //           <button className="btn-sm" onClick={()=>setEditing(e=>!e)}>✏ Rename</button>
// //           {data.type==="manual"&&onDelete&&(
// //             <button className="btn-sm" style={{color:"#f87171",borderColor:"rgba(248,113,113,.3)"}}
// //               onClick={()=>{ if(window.confirm(`Delete album "${data.title}"? Photos are kept.`)){onDelete(data.id);onClose();} }}>
// //               🗑 Delete
// //             </button>
// //           )}
// //           <button className="lb-hbtn lb-close" onClick={onClose}><X size={17}/></button>
// //         </div>
// //       </div>
// //       {editing&&(
// //         <div className="rename-row" style={{marginBottom:12}}>
// //           <input className="rename-in" value={title} onChange={e=>setTitle(e.target.value)}
// //             onKeyDown={e=>e.key==="Enter"&&saveRename()} placeholder="Album name…" autoFocus/>
// //           <input className="rename-in" value={desc} onChange={e=>setDesc(e.target.value)}
// //             placeholder="Description (optional)…" style={{marginLeft:6}}/>
// //           <button className="btn-sm btn-sm--primary" onClick={saveRename} style={{marginLeft:6}}>Save</button>
// //           <button className="btn-sm" onClick={()=>setEditing(false)} style={{marginLeft:4}}>Cancel</button>
// //         </div>
// //       )}
// //       {!editing&&data.description&&(
// //         <div className="album-desc"><span className="album-desc-lbl">DESCRIPTION</span><p>{data.description}</p></div>
// //       )}
// //       {data.images===null
// //         ? <div className="loader-wrap" style={{minHeight:140}}><div className="loader-ring"/></div>
// //         : imgs.length>0
// //           ? <ImgGrid list={imgs} onOpen={i=>onOpen(imgs,i)} onFav={()=>{}} onDel={()=>{}}/>
// //           : (
// //             <div style={{textAlign:"center",padding:"40px 20px",color:"var(--mu)"}}>
// //               <FolderPlus size={40} strokeWidth={1} style={{marginBottom:12,opacity:.4}}/>
// //               <p style={{fontSize:14,marginBottom:6}}>This album is empty</p>
// //               <p style={{fontSize:12,color:"#444"}}>
// //                 Go to <strong>Timeline</strong> → select photos (checkbox icon) →
// //                 choose this album in the batch bar → click the layers icon
// //               </p>
// //             </div>
// //           )
// //       }
// //     </ModalWrap>
// //   );
// // }

// // function ModalWrap({children,onClose,wide}){
// //   return <motion.div className="modal-bg" initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}} onClick={onClose}><motion.div className="modal-box" style={{maxWidth:wide?960:680}} initial={{scale:.95,y:12}} animate={{scale:1,y:0}} exit={{scale:.95}} onClick={e=>e.stopPropagation()}>{children}</motion.div></motion.div>;
// // }

// // function CoOccurrencePanel({faces, onSearch}) {
// //   const [selected, setSelected] = React.useState([]);
// //   const toggle = id => setSelected(p => p.includes(id) ? p.filter(x=>x!==id) : [...p, id]);
// //   return (
// //     <div className="create-album-form">
// //       <p style={{fontSize:12,color:"var(--mu)",marginBottom:10}}>Select 2+ people to find photos where they all appear together:</p>
// //       <div style={{display:"flex",flexWrap:"wrap",gap:8,marginBottom:14,maxHeight:240,overflowY:"auto"}}>
// //         {(faces||[]).map(p=>(
// //           <button key={p.id}
// //             className={"person-chip" + (selected.includes(p.id)?" person-chip--on":"")}
// //             onClick={()=>toggle(p.id)}>
// //             {p.cover && <img src={imgUrl(p.cover)} alt="" style={{width:24,height:24,borderRadius:"50%",objectFit:"cover",marginRight:6}}/>}
// //             {p.name||`Person ${p.id}`}
// //           </button>
// //         ))}
// //         {(!faces||faces.length===0) && <p style={{fontSize:12,color:"var(--mu)"}}>No named people yet. Go to People section first.</p>}
// //       </div>
// //       <button className="btn-sm btn-sm--primary" disabled={selected.length<2} onClick={()=>onSearch(selected)}>
// //         <GitMerge size={12}/> Find Co-appearing Photos ({selected.length} selected)
// //       </button>
// //     </div>
// //   );
// // }

// // function EmotionTimelineChart() {
// //   const [data, setData] = React.useState(null);
// //   React.useEffect(()=>{
// //     axios.get(`${API}/emotion-timeline`).then(r=>setData(r.data)).catch(()=>{});
// //   },[]);

// //   if (!data) return <p style={{fontSize:12,color:"var(--mu)",padding:8}}>Loading…</p>;
// //   if (!data.months?.length) return (
// //     <div style={{padding:"12px 0"}}>
// //       <p style={{fontSize:12,color:"var(--mu)"}}>No emotion timeline data yet.</p>
// //       <p style={{fontSize:11,color:"#555",marginTop:4}}>
// //         Emotions are detected in the background after uploading. Install <code style={{background:"var(--s3)",padding:"1px 5px",borderRadius:3}}>pip install fer</code> for best results, then click <strong>Re-index AI</strong>.
// //       </p>
// //     </div>
// //   );

// //   // Show server hint if only neutral detected
// //   const serverHint = data.hint;

// //   const EMO_COL = {happy:"#4ade80",sad:"#60a5fa",angry:"#f87171",neutral:"#a8a29e",surprised:"#fbbf24",disgusted:"#a78bfa",fearful:"#f9a8d4"};
// //   const EMO_EMOJI = {happy:"😊",sad:"😢",angry:"😠",neutral:"😐",surprised:"😲",disgusted:"🤢",fearful:"😨"};

// //   const months = data.months;
// //   const n = months.length;
// //   // Filter to emotions that actually have data (exclude neutral-only if others exist)
// //   const showEmos = data.emotions.filter(e => (data.series[e]||[]).some(v=>v>0));
// //   const hasRealData = showEmos.some(e => e !== "neutral");

// //   // Chart dimensions — extra bottom padding for legend
// //   const W=560, CHART_H=160, LEG_H=28, PAD={l:36,r:16,t:12,b:32};
// //   const chartW = W - PAD.l - PAD.r;
// //   const chartH = CHART_H - PAD.t - PAD.b;
// //   const maxVal = Math.max(1, ...showEmos.flatMap(e => data.series[e]||[]));

// //   const xScale = i => PAD.l + (n <= 1 ? chartW/2 : (i / (n-1)) * chartW);
// //   const yScale = v => PAD.t + chartH - (v / maxVal) * chartH;
// //   const makePath = series => series.map((v,i) =>
// //     `${i===0?"M":"L"}${xScale(i).toFixed(1)},${yScale(v).toFixed(1)}`
// //   ).join(" ");

// //   // X labels: show at most 8, evenly spaced
// //   const maxLabels = Math.min(8, n);
// //   const labelIndices = n <= maxLabels
// //     ? months.map((_,i)=>i)
// //     : Array.from({length:maxLabels}, (_,k) => Math.round(k*(n-1)/(maxLabels-1)));

// //   return (
// //     <div style={{overflowX:"auto"}}>
// //       {(serverHint || !hasRealData) && (
// //         <p style={{fontSize:11,color:"#f59e0b",marginBottom:8,background:"rgba(245,158,11,.08)",padding:"6px 10px",borderRadius:6,border:"1px solid rgba(245,158,11,.2)"}}>
// //           ⚠️ {serverHint || 'Only "neutral" detected — run pip install fer then Re-index AI.'}
// //         </p>
// //       )}
// //       <svg viewBox={`0 0 ${W} ${CHART_H + LEG_H}`} style={{width:"100%",maxWidth:W,display:"block"}}>
// //         {/* Y grid + labels */}
// //         {[0,0.25,0.5,0.75,1].map((f,i)=>{
// //           const y = PAD.t + chartH*(1-f);
// //           return (
// //             <g key={i}>
// //               <line x1={PAD.l} x2={W-PAD.r} y1={y} y2={y} stroke="#ffffff08" strokeWidth={1}/>
// //               <text x={PAD.l-4} y={y+3} fontSize={7} fill="#555" textAnchor="end">{Math.round(maxVal*f)}</text>
// //             </g>
// //           );
// //         })}
// //         {/* Axes */}
// //         <line x1={PAD.l} x2={PAD.l} y1={PAD.t} y2={PAD.t+chartH} stroke="#333" strokeWidth={1}/>
// //         <line x1={PAD.l} x2={W-PAD.r} y1={PAD.t+chartH} y2={PAD.t+chartH} stroke="#333" strokeWidth={1}/>
// //         {/* Emotion lines + dots */}
// //         {showEmos.map(e=>(
// //           <g key={e}>
// //             <path d={makePath(data.series[e]||[])} fill="none" stroke={EMO_COL[e]||"#888"} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round"/>
// //             {(data.series[e]||[]).map((v,i)=> v>0 && (
// //               <circle key={i} cx={xScale(i)} cy={yScale(v)} r={3} fill={EMO_COL[e]||"#888"} opacity={0.85}/>
// //             ))}
// //           </g>
// //         ))}
// //         {/* X axis labels */}
// //         {labelIndices.map(i=>(
// //           <text key={i} x={xScale(i)} y={PAD.t+chartH+14} fontSize={7} fill="#555" textAnchor="middle">
// //             {months[i]}
// //           </text>
// //         ))}
// //         {/* Legend — below chart, wrapping row */}
// //         {showEmos.map((e,i)=>{
// //           const cols = Math.min(showEmos.length, 4);
// //           const lx = PAD.l + (i % cols) * 130;
// //           const ly = CHART_H + Math.floor(i/cols)*14 + 4;
// //           return (
// //             <g key={e} transform={`translate(${lx},${ly})`}>
// //               <rect width={8} height={8} fill={EMO_COL[e]||"#888"} rx={2}/>
// //               <text x={11} y={7} fontSize={9} fill="#aaa">{EMO_EMOJI[e]} {e}</text>
// //             </g>
// //           );
// //         })}
// //       </svg>
// //     </div>
// //   );
// // }

// // function PeopleFreqChart({people}) {
// //   if(!people?.length) return <p style={{fontSize:12,color:"var(--mu)",padding:8}}>No people data yet.</p>;
// //   const top = people.slice(0,12);
// //   const maxN = Math.max(...top.map(p=>p.count),1);
// //   return (
// //     <div style={{display:"flex",flexDirection:"column",gap:6}}>
// //       {top.map((p,i)=>(
// //         <div key={p.id} style={{display:"flex",alignItems:"center",gap:8}}>
// //           <span style={{fontSize:11,color:"var(--mu)",width:20,textAlign:"right"}}>{i+1}</span>
// //           <span style={{fontSize:12,color:"var(--tx)",width:120,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{p.name||`Person ${p.id}`}</span>
// //           <div style={{flex:1,height:14,background:"var(--s3)",borderRadius:4,overflow:"hidden"}}>
// //             <div style={{width:`${(p.count/maxN)*100}%`,height:"100%",background:"linear-gradient(90deg,#818cf8,#a78bfa)",borderRadius:4,transition:"width .4s"}}/>
// //           </div>
// //           <span style={{fontSize:11,color:"#818cf8",width:30,textAlign:"right"}}>{p.count}</span>
// //         </div>
// //       ))}
// //     </div>
// //   );
// // }

// // function StatsPage({stats, peopleFreq}){
// //   if(!stats) return <Empty icon={BarChart3} msg="Loading stats…"/>;
// //   const cards=[{l:"Photos",v:stats.total_images,c:"#818cf8"},{l:"Faces",v:stats.total_faces,c:"#a78bfa"},{l:"People",v:stats.total_people,c:"#f472b6"},{l:"Albums",v:stats.total_albums,c:"#fbbf24"},{l:"Favorites",v:stats.total_favorites,c:"#4ade80"},{l:"Indexed",v:stats.indexed_vectors,c:"#22d3ee"}];
// //   return (
// //     <div className="stats-wrap">
// //       <PageHead title="Statistics"/>
// //       <div className="stats-cards">{cards.map((c,i)=><motion.div key={i} className="stat-card" whileHover={{y:-3}} style={{"--sc":c.c}}><p className="stat-lbl">{c.l}</p><p className="stat-val" style={{color:c.c}}>{c.v??0}</p></motion.div>)}</div>

// //       <div className="stats-sec">
// //         <p className="stats-sec-lbl">EMOTION TIMELINE</p>
// //         <EmotionTimelineChart/>
// //       </div>

// //       {peopleFreq?.length>0&&(
// //         <div className="stats-sec">
// //           <p className="stats-sec-lbl">MOST PHOTOGRAPHED PEOPLE</p>
// //           <PeopleFreqChart people={peopleFreq}/>
// //         </div>
// //       )}

// //       {stats.color_distribution?.length>0&&<div className="stats-sec"><p className="stats-sec-lbl">COLOR DISTRIBUTION</p><div className="clr-dist">{stats.color_distribution.map((d,i)=><div key={i} className="clr-dist-item"><div className="clr-dist-dot" style={{background:d.color==="gray"?"#888":d.color==="white"?"#ddd":d.color}}/><span>{d.color}</span><span className="clr-dist-n">×{d.count}</span></div>)}</div></div>}
// //       {stats.top_user_tags?.length>0&&<div className="stats-sec"><p className="stats-sec-lbl">YOUR TAGS</p><div className="tag-cloud">{stats.top_user_tags.map((t,i)=><span key={i} className="tag-cloud-item" style={{borderColor:"#86efac44",color:"#86efac"}}>#{t.tag}<span className="tag-n" style={{color:"#86efac99"}}> ×{t.count}</span></span>)}</div></div>}
// //       {stats.top_tags?.length>0&&<div className="stats-sec"><p className="stats-sec-lbl">AI DETECTED OBJECTS</p><div className="tag-cloud">{stats.top_tags.map((t,i)=><span key={i} className="tag-cloud-item">{t.tag}<span className="tag-n"> ×{t.count}</span></span>)}</div></div>}
// //     </div>
// //   );
// // }

// // function PageHead({title,count,unit="photos",extra}){ return <div className="page-head"><h1 className="page-title">{title}</h1>{count>0&&<span className="page-count">{count} {unit}</span>}{extra&&<div style={{marginLeft:"auto"}}>{extra}</div>}</div>; }
// // function Empty({icon:Icon,msg,sub,onClick}){ return <div className={`empty ${onClick?"empty--click":""}`} onClick={onClick}><Icon size={44} strokeWidth={1} color="#333"/><p className="empty-msg">{msg}</p>{sub&&<p className="empty-sub">{sub}</p>}{onClick&&<p className="empty-cta">CLICK TO UPLOAD</p>}</div>; }
// // function Fade({children}){ return <motion.div initial={{opacity:0,y:6}} animate={{opacity:1,y:0}} exit={{opacity:0}} transition={{duration:.18}}>{children}</motion.div>; }

// // /* ══════════════════════════════════════════════════════════════════════════ */
// // /* CSS                                                                         */
// // /* ══════════════════════════════════════════════════════════════════════════ */
// // function CSS() {
// //   return <style>{`
// // @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');
// // *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
// // :root{
// //   --bg:#08080d;--s1:#0f0f16;--s2:#14141d;--s3:#1a1a26;
// //   --br:rgba(255,255,255,.07);--brh:rgba(255,255,255,.13);
// //   --ac:#7c6af7;--ac2:#a99ff7;--aclo:rgba(124,106,247,.13);
// //   --rose:#f43f5e;--gold:#f59e0b;--tx:#dde0ea;--mu:#4a4a62;
// //   --font:'Outfit',sans-serif;--mono:'JetBrains Mono',monospace;
// //   --r:10px;--r2:16px;
// // }
// // html,body,#root{height:100%;background:var(--bg);color:var(--tx);font-family:var(--font);overflow:hidden}
// // ::-webkit-scrollbar{width:3px;height:3px}
// // ::-webkit-scrollbar-thumb{background:rgba(255,255,255,.09);border-radius:3px}
// // .shell{display:flex;height:100vh;overflow:hidden}

// // /* Sidebar */
// // .sidebar{width:206px;flex-shrink:0;background:var(--s1);border-right:1px solid var(--br);display:flex;flex-direction:column;padding:22px 12px 18px;gap:2px;overflow-y:auto}
// // .brand{display:flex;align-items:center;gap:10px;padding:0 5px;margin-bottom:24px;flex-shrink:0}
// // .brand-mark{width:34px;height:34px;border-radius:10px;flex-shrink:0;background:linear-gradient(140deg,#7c6af7,#b89eff);display:flex;align-items:center;justify-content:center;box-shadow:0 4px 18px rgba(124,106,247,.4);color:#fff}
// // .brand-name{font-size:14px;font-weight:800;letter-spacing:.08em;color:#fff}
// // .brand-sub{font-family:var(--mono);font-size:8px;color:var(--mu);letter-spacing:.1em;margin-top:1px}
// // .nav-list{display:flex;flex-direction:column;gap:2px;flex-shrink:0}
// // .nav-btn{display:flex;align-items:center;gap:9px;width:100%;padding:9px 10px;border-radius:8px;background:none;border:none;font-family:var(--font);font-size:13px;font-weight:600;color:var(--mu);cursor:pointer;text-align:left;transition:all .14s;position:relative}
// // .nav-btn:hover{background:rgba(255,255,255,.04);color:var(--tx)}
// // .nav-btn--on{background:var(--aclo)!important;color:var(--ac2)!important}
// // .nav-btn--trash:hover{color:var(--rose);background:rgba(244,63,94,.07)}
// // .nav-ic{display:flex;flex-shrink:0;opacity:.75}
// // .nav-pip{position:absolute;right:10px;width:5px;height:5px;border-radius:50%;background:var(--ac);box-shadow:0 0 7px var(--ac)}
// // .badge{margin-left:auto;background:var(--rose);color:#fff;font-size:8.5px;font-family:var(--mono);padding:2px 5px;border-radius:4px}
// // .sidebar-sep{height:1px;background:var(--br);margin:8px 0;flex-shrink:0}
// // .sidebar-section-label{display:flex;align-items:center;gap:5px;font-family:var(--mono);font-size:8px;color:var(--mu);letter-spacing:.1em;padding:0 10px;margin-bottom:4px;flex-shrink:0}
// // .tag-nav-list{display:flex;flex-direction:column;gap:1px}
// // .tag-nav-btn{display:flex;align-items:center;justify-content:space-between;width:100%;padding:6px 10px;border-radius:7px;background:none;border:none;font-family:var(--mono);font-size:10px;color:var(--mu);cursor:pointer;transition:all .13s;text-align:left}
// // .tag-nav-btn:hover{background:rgba(134,239,172,.07);color:#86efac}
// // .tag-nav-btn--on{background:rgba(134,239,172,.1);color:#86efac}
// // .tag-nav-count{font-size:9px;opacity:.6}
// // .sidebar-foot{display:flex;flex-direction:column;gap:8px;margin-top:auto;padding-top:14px;flex-shrink:0}
// // .reindex-btn{display:flex;align-items:center;justify-content:center;gap:7px;background:var(--s2);border:1px solid var(--br);border-radius:8px;padding:9px;font-family:var(--font);font-size:12px;font-weight:700;color:var(--tx);cursor:pointer;transition:all .14s}
// // .reindex-btn:hover{border-color:var(--ac);color:var(--ac2)}
// // .reindex-btn:disabled{opacity:.45;cursor:default}
// // .reindex-msg{font-family:var(--mono);font-size:8.5px;color:var(--mu);text-align:center;letter-spacing:.06em}
// // .offline-pill{display:flex;align-items:center;gap:7px;background:rgba(74,222,128,.07);border:1px solid rgba(74,222,128,.15);border-radius:7px;padding:7px 10px;font-size:11px;font-weight:600;color:rgba(74,222,128,.8)}
// // .dot-green{width:7px;height:7px;border-radius:50%;background:#4ade80;box-shadow:0 0 7px #4ade80;flex-shrink:0}

// // /* Main */
// // .main{flex:1;display:flex;flex-direction:column;min-width:0;overflow:hidden;position:relative}

// // /* Topbar */
// // .topbar{background:var(--s1);border-bottom:1px solid var(--br);padding:13px 22px;display:flex;flex-direction:column;gap:10px;flex-shrink:0}
// // .mode-strip{display:flex;gap:3px;align-items:center;flex-wrap:wrap}
// // .mode-btn{display:flex;align-items:center;gap:5px;padding:5px 11px;border-radius:6px;border:none;cursor:pointer;font-family:var(--font);font-size:11px;font-weight:700;letter-spacing:.03em;background:none;color:var(--mu);transition:all .13s}
// // .mode-btn:hover{background:rgba(255,255,255,.05);color:var(--tx)}
// // .mode-btn--on{background:var(--ac);color:#fff}
// // .voice-chip{margin-left:8px;display:flex;align-items:center;gap:5px;font-family:var(--mono);font-size:9px;letter-spacing:.07em;padding:4px 10px;border-radius:6px}
// // .voice-chip--rec{background:rgba(244,63,94,.1);border:1px solid rgba(244,63,94,.25);color:var(--rose)}
// // .voice-chip--processing{background:var(--aclo);border:1px solid rgba(124,106,247,.25);color:var(--ac2)}
// // .voice-chip--error{background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.25);color:#f87171}
// // .search-bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
// // .search-wrap{flex:1;min-width:200px;display:flex;align-items:center;gap:6px;background:var(--s2);border:1px solid var(--br);border-radius:var(--r);padding:0 10px;transition:border-color .14s}
// // .search-wrap:focus-within{border-color:var(--ac)}
// // .search-ic{color:var(--mu);flex-shrink:0}
// // .search-in{flex:1;background:none;border:none;padding:9px 0;font-family:var(--font);font-size:13px;color:var(--tx);outline:none}
// // .search-in::placeholder{color:var(--mu)}
// // .mic-btn{background:none;border:none;cursor:pointer;color:var(--mu);padding:3px;display:flex;transition:color .13s}
// // .mic-btn--rec{color:var(--rose)}
// // .mic-btn--proc{color:var(--ac2)}
// // .img-search-label{display:flex;align-items:center;gap:7px;cursor:pointer;background:var(--s2);border:1px solid var(--br);border-radius:var(--r);padding:8px 12px;font-size:12px;color:var(--mu);flex-shrink:0;transition:border-color .13s}
// // .img-search-label:hover{border-color:var(--brh)}
// // .img-search-thumb{width:22px;height:22px;border-radius:4px;object-fit:cover}
// // .color-strip{display:flex;gap:5px;align-items:center;flex-wrap:wrap}
// // .clr-dot{width:22px;height:22px;border-radius:50%;border:none;cursor:pointer;outline:2px solid transparent;outline-offset:2px;transition:all .13s;flex-shrink:0}
// // .clr-dot:hover{transform:scale(1.18)}
// // .clr-dot--on{outline-color:#fff;transform:scale(1.22)}
// // .btn-go{background:var(--ac);color:#fff;border:none;border-radius:8px;padding:8px 18px;font-family:var(--font);font-size:12px;font-weight:700;cursor:pointer;transition:all .13s;flex-shrink:0}
// // .btn-go:hover{background:#9585f8}
// // .btn-batch{display:flex;align-items:center;gap:6px;background:var(--s2);border:1px solid var(--br);color:var(--mu);border-radius:8px;padding:8px 13px;font-family:var(--font);font-size:12px;font-weight:700;cursor:pointer;transition:all .13s;flex-shrink:0}
// // .btn-batch:hover{border-color:var(--ac);color:var(--ac2)}
// // .btn-batch--on{background:var(--aclo);border-color:var(--ac);color:var(--ac2)}
// // .btn-upload{display:flex;align-items:center;gap:6px;background:var(--s2);border:1px solid var(--br);color:var(--tx);border-radius:8px;padding:8px 14px;font-family:var(--font);font-size:12px;font-weight:700;cursor:pointer;transition:all .13s;flex-shrink:0}
// // .btn-upload:hover{border-color:var(--brh);background:var(--s3)}

// // /* Emotion filter bar */
// // .emo-filter-bar{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
// // .emo-filter-label{font-family:var(--mono);font-size:9px;color:var(--mu);letter-spacing:.08em;flex-shrink:0}
// // .emo-filter-btn{display:flex;align-items:center;gap:4px;padding:4px 10px;border-radius:6px;border:1px solid var(--br);background:none;color:var(--mu);font-family:var(--font);font-size:11px;font-weight:600;cursor:pointer;transition:all .13s;flex-shrink:0}
// // .emo-filter-btn:hover{border-color:var(--brh);color:var(--tx)}
// // .emo-filter-btn--on{}
// // .emo-filter-count{font-family:var(--mono);font-size:8px;opacity:.6;margin-left:2px}
// // .emo-filter-clear{display:flex;align-items:center;gap:4px;padding:4px 8px;border-radius:6px;border:1px solid rgba(244,63,94,.3);background:rgba(244,63,94,.08);color:var(--rose);font-family:var(--font);font-size:11px;font-weight:600;cursor:pointer;transition:all .13s}
// // .emo-filter-clear:hover{background:rgba(244,63,94,.15)}

// // /* Body */
// // .body{flex:1;overflow-y:auto;padding:22px 22px 80px}
// // .page-head{display:flex;align-items:baseline;gap:10px;margin-bottom:20px;flex-wrap:wrap}
// // .page-title{font-size:22px;font-weight:800;letter-spacing:-.025em}
// // .page-count{font-family:var(--mono);font-size:9px;color:var(--mu);letter-spacing:.08em}
// // .sel-count{font-family:var(--mono);font-size:9px;color:var(--ac2);background:var(--aclo);padding:2px 8px;border-radius:4px}

// // /* On This Day */
// // .otd-banner{background:linear-gradient(135deg,rgba(245,158,11,.08),rgba(245,158,11,.04));border:1px solid rgba(245,158,11,.2);border-radius:var(--r2);margin-bottom:20px;overflow:hidden}
// // .otd-header{display:flex;align-items:center;justify-content:space-between;padding:12px 16px;cursor:pointer}
// // .otd-left{display:flex;align-items:center;gap:10px}
// // .otd-title{font-size:13px;font-weight:700;display:block}
// // .otd-sub{font-family:var(--mono);font-size:9px;color:var(--mu);display:block;margin-top:2px}
// // .otd-toggle{background:none;border:none;cursor:pointer;color:var(--mu);display:flex;padding:4px}
// // .otd-year{padding:0 16px 14px}
// // .otd-year-label{font-family:var(--mono);font-size:9px;color:rgba(245,158,11,.7);letter-spacing:.08em;margin-bottom:9px;display:block}
// // .otd-strip{display:flex;gap:8px;overflow-x:auto;padding-bottom:4px}
// // .otd-thumb{flex-shrink:0;width:100px;cursor:pointer;border-radius:8px;overflow:hidden;border:1px solid var(--br);transition:border-color .15s}
// // .otd-thumb:hover{border-color:rgba(245,158,11,.4)}
// // .otd-thumb img{width:100%;height:68px;object-fit:cover;display:block}
// // .otd-thumb-caption{font-size:9px;color:var(--mu);padding:4px 6px;line-height:1.4;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}

// // /* Image grid */
// // .img-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(182px,1fr));gap:9px}
// // .img-grid--sm{grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:7px}

// // /* Photo card */
// // .photo-card{position:relative;aspect-ratio:1;border-radius:var(--r);overflow:hidden;cursor:pointer;background:var(--s2);border:1px solid var(--br);transition:border-color .2s,box-shadow .2s}
// // .photo-card:hover{border-color:rgba(124,106,247,.45);box-shadow:0 6px 26px rgba(0,0,0,.5)}
// // .photo-card--sel{border-color:var(--ac)!important;box-shadow:0 0 0 2px var(--ac)!important}
// // .photo-img{width:100%;height:100%;object-fit:cover;transition:transform .25s;display:block}
// // .photo-card:hover .photo-img{transform:scale(1.05)}
// // .chip{position:absolute;font-family:var(--mono);font-size:8.5px;border-radius:4px;border:1px solid;padding:1px 5px;pointer-events:none}
// // .chip--q{top:7px;left:7px}
// // .chip--emo{top:7px;right:7px;font-size:13px;border:none;background:none;padding:0}
// // .fav-dot{position:absolute;bottom:7px;right:7px;color:var(--rose);font-size:13px;pointer-events:none}
// // .batch-check{position:absolute;top:8px;left:8px;filter:drop-shadow(0 1px 3px rgba(0,0,0,.6))}
// // .photo-hover{position:absolute;inset:0;opacity:0;transition:opacity .2s;background:linear-gradient(to top,rgba(0,0,0,.92) 0%,rgba(0,0,0,.35) 45%,transparent 68%);display:flex;flex-direction:column;justify-content:flex-end}
// // .photo-card:hover .photo-hover{opacity:1}
// // .photo-actions{position:absolute;top:7px;right:7px;display:flex;gap:5px}
// // .ph-btn{width:27px;height:27px;border-radius:50%;border:none;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all .14s}
// // .ph-btn:hover{transform:scale(1.15)}
// // .ph-btn--fav{background:rgba(244,63,94,.75);color:#fff}
// // .ph-btn--del{background:rgba(10,10,18,.8);color:#888}
// // .ph-btn--del:hover{background:rgba(220,50,50,.8);color:#fff}
// // .photo-info{padding:8px 8px 5px}
// // .photo-caption{font-size:10px;line-height:1.48;color:rgba(255,255,255,.93);margin-bottom:5px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
// // .photo-meta-row{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:4px}
// // .meta-chip{font-family:var(--mono);font-size:8px;color:rgba(255,255,255,.55);background:rgba(0,0,0,.4);padding:1px 5px;border-radius:3px;display:flex;align-items:center;gap:2px;text-transform:capitalize}
// // .view-hint{display:flex;align-items:center;gap:3px;justify-content:center;padding:4px 0 2px;font-family:var(--mono);font-size:7.5px;color:rgba(255,255,255,.3);letter-spacing:.07em;border-top:1px solid rgba(255,255,255,.06);margin-top:2px}

// // /* Batch bar */
// // .batch-bar{position:absolute;bottom:0;left:0;right:0;background:rgba(15,15,22,.96);backdrop-filter:blur(10px);border-top:1px solid var(--br);padding:12px 22px;display:flex;align-items:center;gap:12px;flex-wrap:wrap;z-index:50}
// // .batch-bar-left{display:flex;align-items:center;gap:8px}
// // .batch-count{display:flex;align-items:center;gap:6px;font-size:13px;font-weight:700;color:var(--ac2)}
// // .batch-sm{background:var(--s3);border:1px solid var(--br);color:var(--mu);border-radius:6px;padding:4px 10px;font-family:var(--font);font-size:11px;font-weight:600;cursor:pointer;transition:all .13s}
// // .batch-sm:hover{border-color:var(--brh);color:var(--tx)}
// // .batch-bar-actions{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-left:auto}
// // .batch-action-btn{display:flex;align-items:center;gap:5px;padding:7px 13px;border-radius:8px;border:1px solid var(--br);background:var(--s2);color:var(--tx);font-family:var(--font);font-size:11px;font-weight:700;cursor:pointer;transition:all .13s}
// // .batch-action-btn:hover{border-color:var(--brh);background:var(--s3)}
// // .batch-action-btn--fav:hover{border-color:rgba(244,63,94,.4);color:var(--rose)}
// // .batch-action-btn--del:hover{border-color:rgba(239,68,68,.4);color:#f87171}
// // .batch-input-group{display:flex;gap:4px;align-items:center}
// // .batch-input{background:var(--s2);border:1px solid var(--br);border-radius:7px;padding:6px 10px;font-family:var(--font);font-size:12px;color:var(--tx);outline:none;width:110px}
// // .batch-input:focus{border-color:var(--ac)}
// // .batch-select{background:var(--s2);border:1px solid var(--br);border-radius:7px;padding:6px 10px;font-family:var(--font);font-size:12px;color:var(--tx);outline:none;cursor:pointer}
// // .batch-select:focus{border-color:var(--ac)}

// // /* Tags in lightbox */
// // .tag-chips{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:8px}
// // .tag-chip-item{display:flex;align-items:center;gap:3px;background:rgba(134,239,172,.1);border:1px solid rgba(134,239,172,.2);color:#86efac;font-family:var(--mono);font-size:9px;padding:2px 7px;border-radius:4px}
// // .tag-chip-x{background:none;border:none;cursor:pointer;color:rgba(134,239,172,.6);display:flex;padding:0;margin-left:1px}
// // .tag-chip-x:hover{color:#86efac}
// // .tag-add-row{display:flex;gap:5px;align-items:center}
// // .tag-add-input{flex:1;background:var(--s3);border:1px solid var(--br);border-radius:6px;padding:5px 9px;font-family:var(--mono);font-size:10px;color:var(--tx);outline:none}
// // .tag-add-input:focus{border-color:var(--ac)}
// // .tag-add-btn{background:var(--s3);border:1px solid var(--br);border-radius:6px;padding:5px 8px;cursor:pointer;color:var(--mu);display:flex;transition:all .13s}
// // .tag-add-btn:hover{border-color:var(--ac);color:var(--ac2)}

// // /* Trash */
// // .trash-card{position:relative;aspect-ratio:1;border-radius:var(--r);overflow:hidden;background:var(--s2);border:1px solid var(--br)}
// // .trash-img{width:100%;height:100%;object-fit:cover;filter:grayscale(.4) brightness(.5)}
// // .trash-overlay{position:absolute;inset:0;opacity:0;transition:opacity .2s;background:rgba(0,0,0,.65);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:6px;padding:10px}
// // .trash-card:hover .trash-overlay{opacity:1}
// // .trash-btn{width:100%;display:flex;align-items:center;justify-content:center;gap:5px;border:none;border-radius:7px;padding:7px;font-family:var(--font);font-size:11px;font-weight:700;cursor:pointer;transition:all .13s}
// // .trash-btn--restore{background:var(--ac);color:#fff}
// // .trash-btn--del{background:rgba(220,50,50,.8);color:#fff}
// // .trash-date{position:absolute;top:7px;left:7px;font-family:var(--mono);font-size:8px;background:rgba(0,0,0,.7);color:#999;padding:2px 6px;border-radius:4px}

// // /* People */
// // .people-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:20px}
// // .person-card{display:flex;flex-direction:column;align-items:center;gap:8px;cursor:pointer}
// // .person-avatar{width:72px;height:72px;border-radius:50%;overflow:hidden;border:2px solid var(--br);background:var(--s2);display:flex;align-items:center;justify-content:center;transition:border-color .2s,box-shadow .2s}
// // .person-avatar img{width:100%;height:100%;object-fit:cover}
// // .person-card:hover .person-avatar{border-color:var(--ac);box-shadow:0 0 18px rgba(124,106,247,.3)}
// // .person-name{font-size:12px;font-weight:700;text-align:center}
// // .person-count{font-family:var(--mono);font-size:8.5px;color:var(--mu)}

// // /* Albums */
// // .albums-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px}
// // .album-card{position:relative;aspect-ratio:16/10;border-radius:var(--r2);overflow:hidden;cursor:pointer;background:var(--s2);border:1px solid var(--br);box-shadow:0 3px 14px rgba(0,0,0,.35);transition:border-color .2s,box-shadow .2s}
// // .album-card:hover{border-color:rgba(124,106,247,.4);box-shadow:0 7px 28px rgba(0,0,0,.55)}
// // .album-cover{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;transition:transform .3s}
// // .album-card:hover .album-cover{transform:scale(1.04)}
// // .album-cover-ph{position:absolute;inset:0;display:flex;align-items:center;justify-content:center}
// // .album-strip{position:absolute;bottom:0;left:0;right:0;height:30px;display:flex;opacity:0;transition:opacity .2s;z-index:2}
// // .album-card:hover .album-strip{opacity:1}
// // .album-strip-img{flex:1;height:100%;object-fit:cover;border-right:1px solid rgba(0,0,0,.3)}
// // .album-info{position:absolute;inset:0;z-index:1;background:linear-gradient(to top,rgba(0,0,0,.92) 0%,rgba(0,0,0,.08) 50%,transparent 75%);padding:12px;display:flex;flex-direction:column;justify-content:flex-end}
// // .album-title{font-weight:800;font-size:14px;letter-spacing:-.01em;margin-bottom:6px;line-height:1.3}
// // .album-meta{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
// // .album-del-btn{position:absolute;top:9px;right:9px;width:26px;height:26px;border-radius:50%;background:rgba(0,0,0,.7);border:1px solid rgba(255,255,255,.15);color:#aaa;display:flex;align-items:center;justify-content:center;cursor:pointer;z-index:3;opacity:0;transition:all .15s}
// // .album-card:hover .album-del-btn{opacity:1}
// // .album-del-btn:hover{background:rgba(220,50,50,.8);color:#fff;border-color:transparent}
// // .tag-sm{font-family:var(--mono);font-size:8.5px;background:rgba(255,255,255,.07);color:var(--mu);padding:2px 7px;border-radius:4px;letter-spacing:.04em}
// // .tag-sm--accent{background:rgba(124,106,247,.18);color:var(--ac2)}
// // .tag-sm--manual{background:rgba(245,158,11,.12);color:rgba(245,158,11,.8)}

// // /* Duplicates */
// // .dupe-list{display:flex;flex-direction:column;gap:16px}
// // .dupe-group{background:var(--s2);border:1px solid var(--br);border-radius:var(--r2);padding:16px}
// // .dupe-head{display:flex;align-items:center;gap:8px;margin-bottom:14px;flex-wrap:wrap}

// // /* Lightbox */
// // .lb-bg{position:fixed;inset:0;z-index:300;background:rgba(0,0,0,.92);backdrop-filter:blur(12px);display:flex;align-items:center;justify-content:center;padding:14px}
// // .lb-box{width:100%;max-width:1160px;max-height:93vh;background:var(--s1);border:1px solid var(--br);border-radius:20px;display:grid;grid-template-columns:1fr 320px;overflow:hidden;box-shadow:0 28px 80px rgba(0,0,0,.75)}
// // .lb-left{position:relative;background:var(--bg);display:flex;align-items:center;justify-content:center;min-height:400px;overflow:hidden}
// // .lb-img{max-width:100%;max-height:90vh;object-fit:contain;display:block}
// // .lb-nav{position:absolute;top:50%;transform:translateY(-50%);width:40px;height:40px;border-radius:50%;border:none;cursor:pointer;background:rgba(255,255,255,.1);backdrop-filter:blur(4px);color:#fff;display:flex;align-items:center;justify-content:center;transition:all .14s;z-index:2}
// // .lb-nav:hover{background:var(--ac);transform:translateY(-50%) scale(1.08)}
// // .lb-nav--l{left:12px}
// // .lb-nav--r{right:12px}
// // .lb-counter{position:absolute;bottom:12px;left:50%;transform:translateX(-50%);font-family:var(--mono);font-size:9.5px;color:rgba(255,255,255,.45);background:rgba(0,0,0,.55);padding:3px 10px;border-radius:20px}
// // .lb-hint{position:absolute;bottom:12px;right:14px;font-family:var(--mono);font-size:7.5px;color:rgba(255,255,255,.2);letter-spacing:.05em}
// // .lb-right{border-left:1px solid var(--br);background:var(--s2);display:flex;flex-direction:column;min-height:0}
// // .lb-right-head{display:flex;align-items:center;justify-content:space-between;padding:15px 17px;border-bottom:1px solid var(--br);flex-shrink:0}
// // .lb-right-title{font-size:14px;font-weight:800;letter-spacing:-.01em}
// // .lb-head-btns{display:flex;align-items:center;gap:4px}
// // .lb-hbtn{background:none;border:none;cursor:pointer;color:var(--mu);padding:6px;border-radius:7px;display:flex;align-items:center;transition:all .13s}
// // .lb-hbtn:hover{background:rgba(255,255,255,.07);color:var(--tx)}
// // .lb-close{color:var(--mu)}
// // .lb-scroll{flex:1;overflow-y:auto;padding:13px 16px;display:flex;flex-direction:column;gap:8px}
// // .db{background:var(--s3);border:1px solid rgba(255,255,255,.05);border-radius:9px;overflow:hidden;border-left:2px solid var(--dc,#7c6af7)}
// // .db-head{display:flex;align-items:center;gap:6px;padding:7px 11px;border-bottom:1px solid rgba(255,255,255,.04)}
// // .db-label{font-family:var(--mono);font-size:8.5px;color:var(--dc,#7c6af7);letter-spacing:.09em;font-weight:500}
// // .db-body{padding:9px 11px}
// // .db-caption{font-size:12.5px;line-height:1.58;color:var(--tx)}
// // .db-caption-sub{font-size:11px;line-height:1.58;color:var(--mu);margin-top:5px}
// // .q-wrap{display:flex;align-items:flex-start;gap:11px}
// // .q-circle{width:44px;height:44px;border-radius:50%;border:2px solid;display:flex;align-items:center;justify-content:center;flex-shrink:0}
// // .q-right{flex:1}
// // .q-level{font-size:12px;font-weight:700;margin-bottom:7px}
// // .qbar-row{display:flex;align-items:center;gap:7px;margin-bottom:4px}
// // .qbar-label{font-family:var(--mono);font-size:8px;color:var(--mu);width:65px;flex-shrink:0}
// // .qbar-track{flex:1;height:3px;background:rgba(255,255,255,.08);border-radius:2px;overflow:hidden}
// // .qbar-fill{height:100%;border-radius:2px}
// // .qbar-num{font-family:var(--mono);font-size:8px;color:var(--mu);width:22px;text-align:right}
// // .aes-row{display:flex;align-items:center;gap:10px}
// // .aes-score{font-size:24px;font-weight:800;color:var(--gold)}
// // .aes-rating{font-size:12px;color:var(--mu)}
// // .emo-row{display:flex;align-items:center;gap:10px}
// // .emo-name{font-size:13px;font-weight:700;text-transform:capitalize}
// // .emo-sub{font-family:var(--mono);font-size:8.5px;color:var(--mu);margin-top:3px}
// // .ocr-text{font-family:var(--mono);font-size:9.5px;color:var(--mu);line-height:1.7;white-space:pre-wrap;max-height:90px;overflow-y:auto}
// // .tag-wrap{display:flex;flex-wrap:wrap;gap:5px}
// // .obj-tag{font-family:var(--mono);font-size:9px;background:rgba(255,255,255,.06);color:var(--mu);padding:2px 8px;border-radius:4px}
// // .meta-stack{display:flex;flex-direction:column;gap:6px}
// // .mrow{display:flex;justify-content:space-between;align-items:baseline;gap:12px}
// // .mrow-k{font-family:var(--mono);font-size:8.5px;color:var(--mu);letter-spacing:.05em;flex-shrink:0}
// // .mrow-v{font-size:11px;color:var(--tx);text-align:right}

// // /* Modal */
// // .modal-bg{position:fixed;inset:0;z-index:200;background:rgba(0,0,0,.83);backdrop-filter:blur(8px);display:flex;align-items:center;justify-content:center;padding:20px}
// // .modal-box{width:100%;max-height:90vh;overflow-y:auto;background:var(--s1);border:1px solid var(--br);border-radius:18px;padding:22px;box-shadow:0 20px 60px rgba(0,0,0,.65)}
// // .modal-head{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:14px}
// // .modal-title{font-size:19px;font-weight:800;letter-spacing:-.02em}
// // .modal-sub{font-family:var(--mono);font-size:8.5px;color:var(--mu);letter-spacing:.08em;margin-top:3px}
// // .rename-row{display:flex;gap:7px;margin-bottom:16px;align-items:center}
// // .rename-in{flex:1;background:var(--s2);border:1px solid var(--br);border-radius:8px;padding:8px 11px;font-family:var(--font);font-size:13px;color:var(--tx);outline:none}
// // .rename-in:focus{border-color:var(--ac)}
// // .album-desc{background:var(--aclo);border:1px solid rgba(124,106,247,.2);border-radius:9px;padding:11px 13px;margin-bottom:16px}
// // .album-desc-lbl{font-family:var(--mono);font-size:8px;color:var(--ac2);letter-spacing:.09em;display:block;margin-bottom:4px}
// // .album-desc p{font-size:12.5px;line-height:1.6}
// // .create-album-form{display:flex;flex-direction:column;gap:10px}
// // .form-input{background:var(--s2);border:1px solid var(--br);border-radius:9px;padding:10px 13px;font-family:var(--font);font-size:13px;color:var(--tx);outline:none}
// // .form-input:focus{border-color:var(--ac)}
// // .form-textarea{background:var(--s2);border:1px solid var(--br);border-radius:9px;padding:10px 13px;font-family:var(--font);font-size:13px;color:var(--tx);outline:none;resize:vertical}
// // .form-textarea:focus{border-color:var(--ac)}
// // .btn-sm{display:inline-flex;align-items:center;gap:5px;background:var(--s2);border:1px solid var(--br);color:var(--tx);border-radius:7px;padding:6px 12px;font-family:var(--font);font-size:11px;font-weight:700;cursor:pointer;transition:all .13s}
// // .btn-sm:hover{border-color:var(--brh);background:var(--s3)}
// // .btn-sm--primary{background:var(--ac);border-color:transparent;color:#fff}
// // .btn-sm--primary:hover{background:#9585f8;border-color:transparent}
// // .btn-sm--accent{background:var(--aclo);border-color:rgba(124,106,247,.25);color:var(--ac2)}
// // .btn-sm--accent:hover{background:var(--ac);color:#fff;border-color:transparent}
// // .btn-sm:disabled{opacity:.45;cursor:default}

// // /* Stats */
// // .stats-wrap{max-width:860px}
// // .stats-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:18px}
// // .stat-card{background:var(--s2);border:1px solid rgba(255,255,255,.05);border-radius:var(--r2);padding:18px 20px;border-top:2px solid var(--sc);transition:transform .14s}
// // .stat-card:hover{transform:translateY(-3px)}
// // .stat-lbl{font-family:var(--mono);font-size:8.5px;color:var(--mu);letter-spacing:.1em;margin-bottom:8px}
// // .stat-val{font-size:40px;font-weight:800;letter-spacing:-.04em;line-height:1}
// // .stats-sec{background:var(--s2);border:1px solid var(--br);border-radius:var(--r2);padding:16px;margin-bottom:11px}
// // .stats-sec-lbl{font-family:var(--mono);font-size:8.5px;color:var(--mu);letter-spacing:.1em;margin-bottom:12px}
// // .clr-dist{display:flex;flex-wrap:wrap;gap:7px}
// // .clr-dist-item{display:flex;align-items:center;gap:6px;background:rgba(255,255,255,.04);border:1px solid var(--br);border-radius:6px;padding:5px 10px;font-size:11px}
// // .clr-dist-dot{width:11px;height:11px;border-radius:50%;border:1px solid rgba(255,255,255,.12);flex-shrink:0}
// // .clr-dist-n{font-family:var(--mono);font-size:9px;color:var(--mu)}
// // .tag-cloud{display:flex;flex-wrap:wrap;gap:6px}
// // .tag-cloud-item{font-family:var(--mono);font-size:9.5px;background:var(--aclo);color:var(--ac2);border:1px solid rgba(124,106,247,.2);padding:3px 10px;border-radius:20px}
// // .tag-n{opacity:.55}

// // /* Empty / loader */
// // .empty{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:260px;background:var(--s2);border:1px dashed var(--brh);border-radius:var(--r2);padding:40px;gap:10px;text-align:center}
// // .empty--click{cursor:pointer;transition:background .14s}
// // .empty--click:hover{background:var(--s3)}
// // .empty-msg{font-size:17px;font-weight:800;letter-spacing:-.02em}
// // .empty-sub{font-size:12.5px;color:var(--mu);max-width:240px}
// // .empty-cta{font-family:var(--mono);font-size:9.5px;color:var(--ac2);letter-spacing:.1em;margin-top:4px}
// // .loader-wrap{display:flex;flex-direction:column;align-items:center;justify-content:center;height:55vh;gap:14px}
// // .loader-ring{width:36px;height:36px;border-radius:50%;border:3px solid rgba(124,106,247,.15);border-top-color:var(--ac);animation:spin .7s linear infinite}
// // .loader-text{font-family:var(--mono);font-size:10px;color:var(--mu);letter-spacing:.12em}
// // @keyframes spin{to{transform:rotate(360deg)}}
// // .spin{animation:spin .7s linear infinite}
// // @media(max-width:720px){.lb-box{grid-template-columns:1fr}.lb-right{border-left:none;border-top:1px solid var(--br);max-height:42vh}.stats-cards{grid-template-columns:repeat(2,1fr)}}
// // /* ── Emoji quick bar ── */
// // .emoji-quick-bar{display:flex;align-items:center;gap:4px;flex-wrap:wrap;padding:2px 0}
// // .emoji-quick-label{font-family:var(--mono);font-size:8px;color:var(--mu);letter-spacing:.08em;flex-shrink:0;margin-right:2px}
// // .emoji-quick-btn{background:none;border:1px solid var(--br);border-radius:6px;padding:3px 6px;font-size:15px;cursor:pointer;transition:all .13s;line-height:1}
// // .emoji-quick-btn:hover{background:var(--aclo);border-color:var(--ac);transform:scale(1.15)}

// // .person-chip{display:flex;align-items:center;padding:5px 10px;background:var(--s2);border:1px solid var(--br);border-radius:20px;font-size:12px;color:var(--mu);cursor:pointer;transition:.15s}
// // .person-chip:hover{border-color:var(--ac);color:var(--tx)}
// // .person-chip--on{background:#312e81;border-color:var(--ac);color:#a5b4fc}
// // `}</style>;
// // }

// import React, { useState, useEffect, useRef, useCallback } from "react";
// import { motion, AnimatePresence } from "framer-motion";
// import {
//   Search, Upload, Image as ImgIcon, Clock, BookOpen, Users, Heart,
//   Copy, Trash2, X, ChevronLeft, ChevronRight, Sparkles, BarChart3,
//   GitMerge, Palette, Shuffle, RotateCcw, AlertTriangle, FileText,
//   Mic, MicOff, Zap, Star, Eye, Camera, Info, Tag, Type, Smile,
//   Loader2, Trash, BookImage, ImageOff, TrendingUp, Aperture,
//   CheckSquare, Square, Plus, Calendar, Hash, FolderPlus,
//   CheckCheck, Layers, ChevronDown, ChevronUp,
//   SlidersHorizontal, UserCheck
// } from "lucide-react";
// import axios from "axios";

// const API = "http://localhost:8000";
// const imgUrl = (f) => {
//   if (!f) return null;
//   const b = String(f).split("/").filter(Boolean).pop();
//   return b ? `${API}/image/${b}` : null;
// };
// const BLANK = `data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='1' height='1'%3E%3Crect fill='%23111118'/%3E%3C/svg%3E`;

// const QL = {
//   Excellent: { c:"#4ade80", bg:"rgba(74,222,128,.12)" },
//   Good:      { c:"#818cf8", bg:"rgba(129,140,248,.12)" },
//   Fair:      { c:"#fbbf24", bg:"rgba(251,191,36,.12)"  },
//   Poor:      { c:"#f87171", bg:"rgba(248,113,113,.12)" },
// };
// const EMO = { happy:"😊",sad:"😢",angry:"😠",surprised:"😲",disgusted:"🤢",fearful:"😨",neutral:"😐" };
// const EMO_COLORS = { happy:"#fbbf24",sad:"#60a5fa",angry:"#f87171",surprised:"#a78bfa",disgusted:"#4ade80",fearful:"#fb923c",neutral:"#6b7280" };
// const qInfo = (l) => QL[l] || { c:"#555", bg:"rgba(255,255,255,.05)" };

// /* ─── Voice hook ─────────────────────────────────────────────────────────── */
// function useVoice(onResult, onError) {
//   const [state, setState] = useState("idle");
//   const streamRef = useRef(null);
//   const processorRef = useRef(null);
//   const ctxRef = useRef(null);
//   const samplesRef = useRef([]);
//   const ok = typeof navigator !== "undefined" && !!navigator.mediaDevices?.getUserMedia;

//   // Convert Float32 PCM samples → 16-bit WAV blob (no ffmpeg needed)
//   const buildWav = (samples, sampleRate) => {
//     const len = samples.length;
//     const buf = new ArrayBuffer(44 + len * 2);
//     const view = new DataView(buf);
//     const write = (off, str) => { for(let i=0;i<str.length;i++) view.setUint8(off+i, str.charCodeAt(i)); };
//     write(0,"RIFF"); view.setUint32(4, 36+len*2, true);
//     write(8,"WAVE"); write(12,"fmt ");
//     view.setUint32(16,16,true); view.setUint16(20,1,true); view.setUint16(22,1,true);
//     view.setUint32(24,sampleRate,true); view.setUint32(28,sampleRate*2,true);
//     view.setUint16(32,2,true); view.setUint16(34,16,true);
//     write(36,"data"); view.setUint32(40,len*2,true);
//     for(let i=0;i<len;i++){
//       const s = Math.max(-1,Math.min(1,samples[i]));
//       view.setInt16(44+i*2, s<0?s*0x8000:s*0x7FFF, true);
//     }
//     return new Blob([buf], {type:"audio/wav"});
//   };

//   const start = useCallback(async () => {
//     if (!ok) return onError?.("Mic unavailable");
//     try {
//       const stream = await navigator.mediaDevices.getUserMedia({
//         audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true }
//       });
//       streamRef.current = stream;
//       samplesRef.current = [];

//       // Use Web Audio API to capture raw PCM at 16kHz — bypasses ffmpeg completely
//       const ctx = new AudioContext({ sampleRate: 16000 });
//       ctxRef.current = ctx;
//       const source = ctx.createMediaStreamSource(stream);
//       const processor = ctx.createScriptProcessor(4096, 1, 1);
//       processorRef.current = processor;

//       processor.onaudioprocess = e => {
//         const data = e.inputBuffer.getChannelData(0);
//         samplesRef.current.push(new Float32Array(data));
//       };
//       source.connect(processor);
//       processor.connect(ctx.destination);
//       setState("rec");
//     } catch(e) {
//       setState("error");
//       onError?.(e.name === "NotAllowedError" ? "Mic permission denied" : "Mic unavailable");
//       setTimeout(()=>setState("idle"),3000);
//     }
//   }, [ok]);

//   const stop = useCallback(async () => {
//     if (state !== "rec") return;
//     setState("processing");
//     try {
//       // Stop all tracks
//       streamRef.current?.getTracks().forEach(t=>t.stop());
//       processorRef.current?.disconnect();
//       await ctxRef.current?.close();

//       // Flatten all captured chunks into one array
//       const allSamples = samplesRef.current;
//       if (!allSamples.length) {
//         setState("error"); onError?.("No audio captured"); setTimeout(()=>setState("idle"),3000); return;
//       }
//       const total = allSamples.reduce((s,c)=>s+c.length,0);
//       const merged = new Float32Array(total);
//       let off=0; for(const c of allSamples){merged.set(c,off);off+=c.length;}

//       // Build WAV and send to Vosk (no ffmpeg needed — already 16kHz mono PCM)
//       const wav = buildWav(merged, 16000);
//       const fd = new FormData(); fd.append("audio", wav, "rec.wav");
//       const {data} = await axios.post(`${API}/voice_search`, fd, {timeout:30000});
//       if (data.success && data.transcript) { setState("idle"); onResult(data.transcript); }
//       else { setState("error"); onError?.(data.error||"No speech detected — speak clearly and try again"); setTimeout(()=>setState("idle"),4000); }
//     } catch(err) {
//       const detail = err?.response?.data?.detail || err.message || "";
//       setState("error");
//       onError?.(err?.response?.status===503 ? "Vosk model not loaded" : detail.slice(0,60) || "Voice failed");
//       setTimeout(()=>setState("idle"),5000);
//     }
//   }, [state]);

//   return { state, ok, start, stop };
// }

// /* ══════════════════════════════════════════════════════════════════════════ */
// export default function App() {
//   const [view,        setView]        = useState("timeline");
//   const [images,      setImages]      = useState([]);
//   const [faces,       setFaces]       = useState([]);
//   const [albums,      setAlbums]      = useState([]);
//   const [dupes,       setDupes]       = useState([]);
//   const [trash,       setTrash]       = useState([]);
//   const [stats,       setStats]       = useState(null);
//   const [loading,     setLoading]     = useState(false);
//   const [searchQ,     setSearchQ]     = useState("");
//   const [searchMode,  setSearchMode]  = useState("text");
//   const [colorPick,   setColorPick]   = useState("blue");
//   const [hybridFile,  setHybridFile]  = useState(null);
//   const [hybridPrev,  setHybridPrev]  = useState(null);
//   const [voiceErr,    setVoiceErr]    = useState("");
//   const [reindexMsg,  setReindexMsg]  = useState("");
//   const [reindexing,  setReindexing]  = useState(false);

//   /* New feature state */
//   const [emoSummary,    setEmoSummary]    = useState([]);
//   const [emoFilter,     setEmoFilter]     = useState("");        // active emotion filter
//   const [onThisDay,     setOnThisDay]     = useState(null);
//   const [otdExpanded,   setOtdExpanded]   = useState(false);

//   const [batchMode,     setBatchMode]     = useState(false);
//   const [selected,      setSelected]      = useState(new Set());
//   const [batchTagInput, setBatchTagInput] = useState("");
//   const [batchAlbumId,  setBatchAlbumId]  = useState("");

//   const [allTags,       setAllTags]       = useState([]);
//   const [activeTag,     setActiveTag]     = useState("");

//   const [createAlbumOpen, setCreateAlbumOpen] = useState(false);
//   const [newAlbumTitle,   setNewAlbumTitle]   = useState("");
//   const [newAlbumDesc,    setNewAlbumDesc]    = useState("");
//   const [pendingAlbumImages, setPendingAlbumImages] = useState([]); // snapshot of selected at open time

//   // Named event creation
//   const [createEventOpen, setCreateEventOpen] = useState(false);
//   const [newEventTitle,   setNewEventTitle]   = useState("");
//   const [newEventType,    setNewEventType]    = useState("Other");
//   const [newEventDate,    setNewEventDate]    = useState("");
//   const [newEventDesc,    setNewEventDesc]    = useState("");

//   // Advanced filters
//   const [minPeople,     setMinPeople]     = useState(0);
//   const [coPersonIds,   setCoPersonIds]   = useState([]);
//   const [showCoModal,   setShowCoModal]   = useState(false);

//   // Face similarity
//   const [faceSrcFile,   setFaceSrcFile]   = useState(null);
//   const [faceSrcPrev,   setFaceSrcPrev]   = useState(null);
//   const [showFaceSim,   setShowFaceSim]   = useState(false);

//   // People frequency
//   const [peopleFreq,    setPeopleFreq]    = useState([]);
//   // People suggestions shown when search returns no results
//   const [peopleSuggestions, setPeopleSuggestions] = useState([]);

//   /* Lightbox */
//   const [lb, setLb] = useState(null);
//   /* Modals */
//   const [personModal, setPersonModal] = useState(null);
//   const [albumModal,  setAlbumModal]  = useState(null);

//   const uploadRef = useRef(null);
//   const voice = useVoice(
//     t => { setSearchQ(t); setSearchMode("text"); setVoiceErr(""); runSearch("text",t); },
//     setVoiceErr
//   );

//   /* keyboard */
//   useEffect(() => {
//     const h = e => {
//       if (!lb) return;
//       if (e.key==="ArrowRight") shiftLb(1);
//       if (e.key==="ArrowLeft")  shiftLb(-1);
//       if (e.key==="Escape")     setLb(null);
//     };
//     window.addEventListener("keydown",h);
//     return ()=>window.removeEventListener("keydown",h);
//   }, [lb]);

//   const shiftLb = useCallback(d=>setLb(p=>p?{...p,idx:(p.idx+d+p.list.length)%p.list.length}:null),[]);
//   const openLb  = (list,idx)=>setLb({list,idx});
//   const closeLb = ()=>setLb(null);

//   /* load */
//   useEffect(()=>{ load(); },[view]);
//   useEffect(()=>{
//     loadEmoSummary(); loadTags(); loadTrashCount(); loadPeopleFreq();

//   }, []);
//   useEffect(()=>{ if(view==="timeline") loadOnThisDay(); }, [view]);

//   const load = async () => {
//     setLoading(true);
//     try {
//       if (view==="timeline")   { const r=await axios.get(`${API}/timeline`);              setImages(r.data.results||[]); }
//       if (view==="favorites")  { const r=await axios.get(`${API}/favorites`);             setImages(r.data.results||[]); }
//       if (view==="explore")    { const r=await axios.get(`${API}/explore/random?count=24`);setImages(r.data.results||[]); }
//       if (view==="faces")      { const r=await axios.get(`${API}/faces`);                 setFaces(r.data.results||[]); }
//       if (view==="albums")     { const r=await axios.get(`${API}/albums`);                setAlbums(r.data.results||[]); }
//       if (view==="duplicates") { const r=await axios.get(`${API}/duplicates`);            setDupes(r.data.duplicate_groups||[]); }
//       if (view==="trash")      { const r=await axios.get(`${API}/trash`);                 setTrash(r.data.results||[]); }
//       if (view==="stats")      { const r=await axios.get(`${API}/stats`);                 setStats(r.data); }
//     } catch {}
//     setSelected(new Set()); setBatchMode(false);
//     setLoading(false);
//   };

//   const loadEmoSummary = async () => {
//     try { const r=await axios.get(`${API}/emotions/summary`); setEmoSummary(r.data.emotions||[]); } catch {}
//   };
//   const loadOnThisDay = async () => {
//     try { const r=await axios.get(`${API}/on-this-day`); setOnThisDay(r.data); } catch {}
//   };
//   const loadTags = async () => {
//     try { const r=await axios.get(`${API}/tags`); setAllTags(r.data.tags||[]); } catch {}
//   };
//   const loadTrashCount = async () => {
//     try { const r=await axios.get(`${API}/trash`); setTrash(r.data.results||[]); } catch {}
//   };

//   const loadPeopleFreq = async () => {
//     try { const r=await axios.get(`${API}/people/frequency`); setPeopleFreq(r.data.people||[]); } catch {}
//   };

//   // Group photo filter
//   const filterGroupPhotos = async (n) => {
//     setMinPeople(n); setLoading(true); setView("search");
//     try { const r=await axios.get(`${API}/group-photos?min_people=${n}`); setImages(r.data.results||[]); }
//     catch {} finally { setLoading(false); }
//   };

//   // Co-occurrence search
//   const runCoOccurrence = async (ids) => {
//     if (!ids||ids.length<2) return;
//     setLoading(true); setView("search"); setShowCoModal(false);
//     try { const r=await axios.get(`${API}/co-occurrence?person_ids=${ids.join(",")}`); setImages(r.data.results||[]); }
//     catch {} finally { setLoading(false); }
//   };

//   // Face similarity
//   const runFaceSimilarity = async () => {
//     if (!faceSrcFile) return;
//     setLoading(true); setView("search"); setShowFaceSim(false);
//     const fd=new FormData(); fd.append("file", faceSrcFile); fd.append("top_k","30");
//     try { const r=await axios.post(`${API}/face-similarity`, fd); setImages(r.data.results||[]); }
//     catch(err) { alert("Face search failed: "+(err?.response?.data?.detail||err.message)); }
//     finally { setLoading(false); setFaceSrcFile(null); setFaceSrcPrev(null); }
//   };

//   const runSearch = async (mode=searchMode, q=searchQ) => {
//     setEmoFilter(""); setActiveTag(""); setLoading(true); setView("search");
//     try {
//       const fd=new FormData(); let res;
//       if (mode==="text")     { fd.append("query",q);          res=await axios.post(`${API}/search`,fd); }
//       if (mode==="describe") { fd.append("description",q);    res=await axios.post(`${API}/search/describe`,fd); }
//       if (mode==="image")    { fd.append("file",hybridFile); fd.append("top_k",20); res=await axios.post(`${API}/search/image`,fd); }
//       if (mode==="hybrid")   { fd.append("query",q); if(hybridFile)fd.append("file",hybridFile); fd.append("top_k",20); res=await axios.post(`${API}/search/hybrid`,fd); }
//       if (mode==="color")    { fd.append("color",colorPick); fd.append("top_k",20); res=await axios.post(`${API}/search/color`,fd); }
//       let results = res?.data?.results || [];
//       // If text search returned nothing, try person name search as fallback
//       if (mode==="text" && results.length===0 && q.trim().length>=2) {
//         try {
//           const pr = await axios.get(`${API}/people/search?q=${encodeURIComponent(q.trim())}`);
//           if (pr.data.results?.length) {
//             results = pr.data.results.flatMap(p => p.results || []);
//           }
//         } catch {}
//       }
//       // Show people suggestions if still no results
//       if (mode==="text" && results.length===0) {
//         const suggestions = res?.data?.people_suggestions || [];
//         setPeopleSuggestions(suggestions);
//       } else {
//         setPeopleSuggestions([]);
//       }
//       setImages(results);
//     } catch { setImages([]); }
//     setLoading(false);
//   };

//   const filterByEmotion = async (emo) => {
//     if (emoFilter===emo) { setEmoFilter(""); load(); return; }
//     setEmoFilter(emo); setView("search"); setLoading(true);
//     try { const fd=new FormData(); fd.append("emotion",emo); const r=await axios.post(`${API}/search/emotion`,fd); setImages(r.data.results||[]); }
//     catch { setImages([]); }
//     setLoading(false);
//   };

//   const filterByTag = async (tag) => {
//     if (activeTag===tag) { setActiveTag(""); load(); return; }
//     setActiveTag(tag); setView("timeline"); setLoading(true);
//     try { const r=await axios.get(`${API}/tags/${tag}/images`); setImages(r.data.results||[]); }
//     catch { setImages([]); }
//     setLoading(false);
//   };

//   const handleUpload = async (files) => {
//     setLoading(true);
//     for (const f of Array.from(files)) { try { const fd=new FormData(); fd.append("file",f); await axios.post(`${API}/upload`,fd); } catch {} }
//     load();
//   };

//   const doReindex = async () => {
//     setReindexing(true); setReindexMsg("Clustering…");
//     try { const r=await axios.post(`${API}/recluster`); setReindexMsg(`✓ ${r.data.people} people · ${r.data.albums} albums`); setTimeout(load,500); }
//     catch { setReindexMsg("Failed"); }
//     setReindexing(false);
//   };

//   const reprocessEmotions = async () => {
//     setReindexing(true); setReindexMsg("Reprocessing emotions…");
//     try {
//       const r = await axios.post(`${API}/reprocess-emotions`);
//       setReindexMsg(`✓ ${r.data.message}`);
//       setTimeout(()=>{ loadEmoSummary(); if(view==="stats") load(); }, 3000);
//     } catch(err) {
//       setReindexMsg("Emotion reprocess failed: " + (err?.response?.data?.detail || err.message));
//     }
//     setReindexing(false);
//   };

//   const reprocessColors = async () => {
//     setReindexing(true); setReindexMsg("Recomputing colors…");
//     try {
//       const r = await axios.post(`${API}/reprocess-colors`);
//       setReindexMsg(`✓ Colors updated for ${r.data.updated} photos`);
//     } catch(err) {
//       setReindexMsg("Color reprocess failed: " + (err?.response?.data?.detail || err.message));
//     }
//     setReindexing(false);
//   };

//   const reprocessNames = async () => {
//     setReindexing(true); setReindexMsg("Auto-naming people…");
//     try {
//       const r = await axios.post(`${API}/reprocess-names`);
//       setReindexMsg(`✓ ${r.data.message}`);
//       setTimeout(()=>{ if(view==="faces") load(); }, 1000);
//     } catch(err) {
//       setReindexMsg("Name detection failed: " + (err?.response?.data?.detail || err.message));
//     }
//     setReindexing(false);
//   };

//   const toggleFav = async (id, e) => {
//     e?.stopPropagation();
//     try { const fd=new FormData(); fd.append("image_id",id); await axios.post(`${API}/toggle_favorite`,fd); if(view==="favorites")load(); } catch {}
//   };

//   const softDelete = async (id, e) => {
//     e?.stopPropagation();
//     if (!window.confirm("Move to trash?")) return;
//     try {
//       const fd=new FormData(); fd.append("image_id",id);
//       await axios.post(`${API}/delete_image`,fd);
//       setImages(p=>p.filter(i=>i.id!==id));
//       loadTrashCount();
//     } catch(err) {
//       alert("Could not move to trash: " + (err?.response?.data?.detail || err.message));
//     }
//   };

//   const restoreImg = async id => {
//     try {
//       const fd=new FormData(); fd.append("image_id",id);
//       await axios.post(`${API}/restore`,fd);
//       setTrash(p=>p.filter(i=>i.id!==id));
//     } catch(err) {
//       alert("Restore failed: " + (err?.response?.data?.detail || err.message));
//     }
//   };
//   const permDelete = async id => {
//     if (!window.confirm("Delete forever? This cannot be undone.")) return;
//     try {
//       const fd=new FormData(); fd.append("image_id",id);
//       await axios.post(`${API}/permanent_delete`,fd);
//       setTrash(p=>p.filter(i=>i.id!==id));
//     } catch(err) {
//       alert("Delete failed: " + (err?.response?.data?.detail || err.message));
//     }
//   };

//   const openPerson = async p => {
//     setPersonModal({...p,images:null});
//     try { const r=await axios.get(`${API}/people/${p.id}`); setPersonModal(r.data); } catch {}
//   };
//   const openAlbum = async a => {
//     setAlbumModal({...a,images:null});
//     try { const r=await axios.get(`${API}/albums/${a.id}`); setAlbumModal(r.data); } catch {}
//   };
//   const renamePerson = async (id,name) => {
//     try { const fd=new FormData(); fd.append("name",name); await axios.post(`${API}/people/${id}`,fd); load(); setPersonModal(p=>p&&{...p,name}); } catch {}
//   };

//   /* Batch actions */
//   const toggleSelect = useCallback((id) => {
//     setSelected(prev => { const s=new Set(prev); s.has(id)?s.delete(id):s.add(id); return s; });
//   }, []);
//   const selectAll = () => setSelected(new Set(images.map(i=>i.id)));
//   const clearSel  = () => setSelected(new Set());

//   const batchFavorite = async (val) => {
//     if (!selected.size) return;
//     const fd=new FormData(); fd.append("image_ids",[...selected].join(",")); fd.append("value",val?1:0);
//     try { await axios.post(`${API}/batch/favorite`,fd); load(); } catch {}
//   };
//   const batchDelete = async () => {
//     if (!selected.size||!window.confirm(`Trash ${selected.size} photos?`)) return;
//     const fd=new FormData(); fd.append("image_ids",[...selected].join(","));
//     try { await axios.post(`${API}/batch/delete`,fd); load(); } catch {}
//   };
//   const batchTag = async () => {
//     if (!selected.size||!batchTagInput.trim()) return;
//     const fd=new FormData(); fd.append("image_ids",[...selected].join(",")); fd.append("tag",batchTagInput.trim());
//     try { await axios.post(`${API}/batch/tag`,fd); setBatchTagInput(""); loadTags(); } catch {}
//   };
//   const batchAddAlbum = async () => {
//     if (!selected.size||!batchAlbumId) return;
//     const fd=new FormData(); fd.append("image_ids",[...selected].join(",")); fd.append("album_id",batchAlbumId);
//     try { await axios.post(`${API}/batch/album`,fd); setBatchAlbumId(""); load(); } catch {}
//   };

//   const createAlbum = async (imageIds=[]) => {
//     if (!newAlbumTitle.trim()) return;
//     const fd=new FormData();
//     fd.append("title", newAlbumTitle);
//     fd.append("description", newAlbumDesc);
//     // Priority: explicit imageIds arg → pendingAlbumImages snapshot → current selected
//     const ids = imageIds.length ? imageIds
//               : pendingAlbumImages.length ? pendingAlbumImages
//               : [...selected];
//     console.log("Creating album with", ids.length, "images:", ids);
//     if (ids.length) fd.append("image_ids", ids.join(","));
//     try {
//       const r = await axios.post(`${API}/albums/create`, fd);
//       setCreateAlbumOpen(false); setNewAlbumTitle(""); setNewAlbumDesc("");
//       const addedCount = ids.length;
//       setPendingAlbumImages([]);
//       setSelected(new Set()); setBatchMode(false);
//       setView("albums");
//       await load();
//       // Open the newly created album immediately so user can see it
//       if (r.data?.id) {
//         const newAlbum = {id: r.data.id, title: r.data.title, type:"manual",
//                          count: addedCount, images: null};
//         openAlbum(newAlbum);
//       }
//       return r.data;
//     } catch(err) { alert("Failed to create album: " + (err?.response?.data?.detail || err.message)); }
//   };
//   const deleteAlbum = async id => {
//     try {
//       await axios.delete(`${API}/albums/${id}/delete`);
//       setAlbumModal(null);
//       load();
//     } catch(err) { alert("Delete failed: " + (err?.response?.data?.detail || err.message)); }
//   };

//   const renameAlbum = async (id, title, description) => {
//     const fd=new FormData(); fd.append("title", title);
//     if (description) fd.append("description", description);
//     try {
//       await axios.post(`${API}/albums/${id}/rename`, fd);
//       load();
//       setAlbumModal(prev => prev && prev.id===id ? {...prev, title, description: description||prev.description} : prev);
//     } catch(err) { alert("Rename failed: " + (err?.response?.data?.detail || err.message)); }
//   };

//   const EVENT_TYPES = ["Birthday","Trip","Vacation","Wedding","Anniversary","Graduation","Party","Holiday","Family","Work","Other"];

//   const createNamedEvent = async () => {
//     if (!newEventTitle.trim()) return;
//     const fd=new FormData();
//     fd.append("title", newEventTitle.trim());
//     fd.append("event_type", newEventType);
//     fd.append("description", newEventDesc.trim());
//     fd.append("date_str", newEventDate);
//     try {
//       await axios.post(`${API}/events/create`, fd);
//       setCreateEventOpen(false);
//       setNewEventTitle(""); setNewEventType("Other"); setNewEventDate(""); setNewEventDesc("");
//       load();
//     } catch(err) { alert("Could not create event: " + (err?.response?.data?.detail || err.message)); }
//   };

//   const addTagToImage = async (imgId, tag) => {
//     const fd=new FormData(); fd.append("tag",tag);
//     try {
//       const r = await axios.post(`${API}/photo/${imgId}/tags/add`,fd);
//       const newTags = r.data.tags || [];
//       // Update the images list in place so lightbox reflects latest tags
//       setImages(prev => prev.map(i => i.id===imgId ? {...i, user_tags: newTags} : i));
//       loadTags();
//       return newTags;
//     } catch { return null; }
//   };
//   const removeTagFromImage = async (imgId, tag) => {
//     const fd=new FormData(); fd.append("tag",tag);
//     try {
//       const r = await axios.post(`${API}/photo/${imgId}/tags/remove`,fd);
//       const newTags = r.data.tags || [];
//       setImages(prev => prev.map(i => i.id===imgId ? {...i, user_tags: newTags} : i));
//       loadTags();
//       return newTags;
//     } catch { return null; }
//   };

//   const isGridView = ["timeline","search","favorites","explore"].includes(view);

//   const NAV = [
//     { id:"timeline",   icon:<Clock size={15}/>,     label:"Timeline"   },
//     { id:"albums",     icon:<BookOpen size={15}/>,  label:"Albums"     },
//     { id:"faces",      icon:<Users size={15}/>,     label:"People"     },
//     { id:"favorites",  icon:<Heart size={15}/>,     label:"Favorites"  },
//     { id:"duplicates", icon:<Copy size={15}/>,      label:"Duplicates" },
//     { id:"explore",    icon:<Shuffle size={15}/>,   label:"Explore"    },
//     { id:"stats",      icon:<BarChart3 size={15}/>, label:"Statistics" },
//   ];
//   const MODES = [
//     { id:"text",     Icon:Search,   label:"Search"   },
//     { id:"describe", Icon:FileText, label:"Describe" },
//     { id:"image",    Icon:ImgIcon,  label:"By Image" },
//     { id:"hybrid",   Icon:GitMerge, label:"Hybrid"   },
//     { id:"color",    Icon:Palette,  label:"By Color" },
//   ];
//   const COLORS = ["red","orange","yellow","green","blue","purple","pink","white","black","gray","brown"];

//   return (
//     <>
//       <CSS />
//       <div className="shell">

//         {/* ── SIDEBAR ─────────────────────────────────────────────────── */}
//         <nav className="sidebar">
//           <div className="brand">
//             <div className="brand-mark"><Aperture size={16} strokeWidth={1.5}/></div>
//             <div>
//               <p className="brand-name">LUMINA</p>
//               <p className="brand-sub">AI Gallery</p>
//             </div>
//           </div>

//           <div className="nav-list">
//             {NAV.map(n=>(
//               <button key={n.id} className={`nav-btn ${view===n.id?"nav-btn--on":""}`} onClick={()=>setView(n.id)}>
//                 <span className="nav-ic">{n.icon}</span>
//                 <span>{n.label}</span>
//                 {view===n.id && <span className="nav-pip"/>}
//               </button>
//             ))}
//           </div>

//           <div className="sidebar-sep"/>

//           {/* Advanced Filters */}
//           <p className="sidebar-section-label"><SlidersHorizontal size={10}/> FILTERS</p>
//           <button className="nav-btn" style={{fontSize:11}} onClick={()=>filterGroupPhotos(2)}>
//             <span className="nav-ic"><Users size={14}/></span><span>2+ People</span>
//           </button>
//           <button className="nav-btn" style={{fontSize:11}} onClick={()=>filterGroupPhotos(3)}>
//             <span className="nav-ic"><Users size={14}/></span><span>3+ People</span>
//           </button>
//           <button className="nav-btn" style={{fontSize:11}} onClick={()=>filterGroupPhotos(5)}>
//             <span className="nav-ic"><Users size={14}/></span><span>5+ People</span>
//           </button>
//           <button className="nav-btn" style={{fontSize:11}} onClick={()=>setShowCoModal(true)}>
//             <span className="nav-ic"><GitMerge size={14}/></span><span>Co-Appear</span>
//           </button>
//           <button className="nav-btn" style={{fontSize:11}} onClick={()=>setShowFaceSim(true)}>
//             <span className="nav-ic"><UserCheck size={14}/></span><span>Face Search</span>
//           </button>

//           <div className="sidebar-sep"/>

//           <button className={`nav-btn nav-btn--trash ${view==="trash"?"nav-btn--on":""}`} onClick={()=>setView("trash")}>
//             <span className="nav-ic"><Trash size={15}/></span>
//             <span>Trash</span>
//             {trash.length>0 && <span className="badge">{trash.length}</span>}
//           </button>

//           {/* Tags sidebar section */}
//           {allTags.length>0 && (
//             <>
//               <div className="sidebar-sep"/>
//               <p className="sidebar-section-label"><Hash size={10}/> TAGS</p>
//               <div className="tag-nav-list">
//                 {allTags.slice(0,8).map(t=>(
//                   <button key={t.tag} className={`tag-nav-btn ${activeTag===t.tag?"tag-nav-btn--on":""}`} onClick={()=>filterByTag(t.tag)}>
//                     <span>#{t.tag}</span><span className="tag-nav-count">{t.count}</span>
//                   </button>
//                 ))}
//               </div>
//             </>
//           )}

//           <div className="sidebar-sep"/>
//           <div className="sidebar-foot">
//             <button className="reindex-btn reindex-btn--primary" onClick={doReindex} disabled={reindexing}>
//               <Zap size={13} className={reindexing?"spin":""}/>{reindexing?"Indexing…":"Re-index AI"}
//             </button>
//             <ToolsAccordion reindexing={reindexing}
//               onEmotions={reprocessEmotions}
//               onColors={reprocessColors}
//               onNames={reprocessNames}
//               onCleanup={async()=>{ if(!window.confirm("Delete all empty albums?")) return; const r=await axios.delete(`${API}/albums/empty/cleanup`); setReindexMsg(`✓ Deleted ${r.data.deleted} empty albums`); load(); }}
//             />
//             {reindexMsg && <p className="reindex-msg">{reindexMsg}</p>}
//             <div className="offline-pill"><span className="dot-green"/><span>Fully Offline</span></div>
//             {voiceErr && voiceErr.includes("setup") && (
//               <div style={{background:"rgba(245,158,11,.08)",border:"1px solid rgba(245,158,11,.2)",borderRadius:7,padding:"8px 10px",fontSize:10,color:"rgba(245,158,11,.9)",lineHeight:1.5}}>
//                 <strong>Voice Setup:</strong><br/>
//                 1. pip install vosk<br/>
//                 2. Download model from<br/>
//                 <span style={{color:"#60a5fa",wordBreak:"break-all"}}>alphacephei.com/vosk/models</span><br/>
//                 3. Extract to:<br/>
//                 <code style={{fontSize:9,color:"#aaa"}}>models/vosk-model-small-en-us</code>
//               </div>
//             )}
//           </div>
//         </nav>

//         {/* ── MAIN ────────────────────────────────────────────────────── */}
//         <div className="main">

//           {/* TOPBAR */}
//           <header className="topbar">
//             <div className="mode-strip">
//               {MODES.map(({id,Icon,label})=>(
//                 <button key={id} className={`mode-btn ${searchMode===id?"mode-btn--on":""}`} onClick={()=>setSearchMode(id)}>
//                   <Icon size={11}/> {label}
//                 </button>
//               ))}
//               {voice.state!=="idle" && (
//                 <span className={`voice-chip voice-chip--${voice.state}`}>
//                   {voice.state==="rec"        && <><MicOff size={10}/> Recording</>}
//                   {voice.state==="processing" && <><Loader2 size={10} className="spin"/> Transcribing…</>}
//                   {voice.state==="error"      && <span title={voiceErr}>{voiceErr.length>38?voiceErr.slice(0,38)+"…":voiceErr}</span>}
//                 </span>
//               )}
//             </div>
//             {/* ── Quick emoji search chips ───────────────────────────────── */}
//             <div className="emoji-quick-bar">
//               <span className="emoji-quick-label">Quick:</span>
//               {[["😊","happy"],["😢","sad"],["😠","angry"],["😲","surprised"],["😨","fearful"],
//                 ["🐶","dog"],["🏖️","beach"],["🌅","sunset"],["🎉","party"],["✈️","travel"],
//                 ["❄️","snow"],["🌸","flowers"],["🍕","food"],["🏋️","gym"],["🌊","ocean"]
//               ].map(([emoji,label])=>(
//                 <button key={emoji} className="emoji-quick-btn" title={label}
//                   onClick={()=>{ setSearchQ(emoji); setSearchMode("text"); runSearch("text", emoji); }}>
//                   {emoji}
//                 </button>
//               ))}
//             </div>
//             <div className="search-bar">
//               {(searchMode==="text"||searchMode==="describe"||searchMode==="hybrid") && (
//                 <div className="search-wrap">
//                   <Search size={14} className="search-ic"/>
//                   <input className="search-in" value={searchQ} onChange={e=>setSearchQ(e.target.value)}
//                     onKeyDown={e=>e.key==="Enter"&&runSearch()}
//                     placeholder={searchMode==="describe"?"Describe what's in the photo…":searchMode==="hybrid"?"Text + optional image…":"Search by scene, object, person, text…"}
//                   />
//                   {voice.ok && (
//                     <button className={`mic-btn ${voice.state==="rec"?"mic-btn--rec":voice.state==="processing"?"mic-btn--proc":""}`}
//                       onClick={voice.state==="rec"?voice.stop:voice.start}>
//                       {voice.state==="rec"?<MicOff size={14}/>:voice.state==="processing"?<Loader2 size={14} className="spin"/>:<Mic size={14}/>}
//                     </button>
//                   )}
//                 </div>
//               )}
//               {(searchMode==="image"||searchMode==="hybrid") && (
//                 <label className="img-search-label">
//                   {hybridPrev?<img src={hybridPrev} className="img-search-thumb" alt=""/>:<ImgIcon size={13}/>}
//                   <span>{hybridPrev?"Ready":"Upload ref"}</span>
//                   <input type="file" accept="image/*" style={{display:"none"}} onChange={e=>{const f=e.target.files?.[0];if(f){setHybridFile(f);setHybridPrev(URL.createObjectURL(f))}}}/>
//                 </label>
//               )}
//               {searchMode==="color" && (
//                 <div className="color-strip">
//                   {COLORS.map(c=>(
//                     <button key={c} title={c} className={`clr-dot ${colorPick===c?"clr-dot--on":""}`}
//                       style={{background:c==="white"?"#e5e5e5":c==="gray"?"#888":c}}
//                       onClick={()=>setColorPick(c)}/>
//                   ))}
//                 </div>
//               )}
//               <button className="btn-go" onClick={()=>runSearch()}>Search</button>
//               {isGridView && (
//                 <button className={`btn-batch ${batchMode?"btn-batch--on":""}`} onClick={()=>{setBatchMode(p=>!p);clearSel();}}>
//                   <CheckSquare size={13}/> {batchMode?"Exit Select":"Select"}
//                 </button>
//               )}
// <button className="btn-upload" onClick={()=>uploadRef.current?.click()}>
//                 <Upload size={13}/> Upload
//               </button>
//               <input ref={uploadRef} type="file" multiple accept="image/*" style={{display:"none"}} onChange={e=>handleUpload(e.target.files)}/>
//             </div>

//             {/* Emotion filter bar */}
//             {isGridView && emoSummary.length>0 && (
//               <div className="emo-filter-bar">
//                 <span className="emo-filter-label">Filter:</span>
//                 {emoSummary.filter(e=>e.emotion&&e.emotion!=="neutral").map(e=>(
//                   <button key={e.emotion}
//                     className={`emo-filter-btn ${emoFilter===e.emotion?"emo-filter-btn--on":""}`}
//                     style={emoFilter===e.emotion?{background:EMO_COLORS[e.emotion]+"22",borderColor:EMO_COLORS[e.emotion]+"66",color:EMO_COLORS[e.emotion]}:{}}
//                     onClick={()=>filterByEmotion(e.emotion)}>
//                     {EMO[e.emotion]||"😐"} {e.emotion} <span className="emo-filter-count">{e.count}</span>
//                   </button>
//                 ))}
//                 {(emoFilter||activeTag) && (
//                   <button className="emo-filter-clear" onClick={()=>{setEmoFilter("");setActiveTag("");load();}}>
//                     <X size={10}/> Clear
//                   </button>
//                 )}
//               </div>
//             )}
//           </header>

//           {/* BODY */}
//           <div className="body">
//             <AnimatePresence mode="wait">
//               {loading ? (
//                 <Fade key="load">
//                   <div className="loader-wrap"><div className="loader-ring"/><p className="loader-text">Loading…</p></div>
//                 </Fade>

//               ) : isGridView ? (
//                 <Fade key={view+emoFilter+activeTag}>
//                   {/* On This Day banner */}
//                   {view==="timeline" && onThisDay?.total>0 && (
//                     <OnThisDayBanner data={onThisDay} expanded={otdExpanded} onToggle={()=>setOtdExpanded(p=>!p)} onOpen={openLb}/>
//                   )}

//                   <div className="page-head">
//                     <h1 className="page-title">
//                       {view==="search"?`Results for "${searchQ}"`:view==="favorites"?"Favorites":view==="explore"?"Explore":
//                        activeTag?`#${activeTag}`:emoFilter?`${EMO[emoFilter]} ${emoFilter}`:"Timeline"}
//                     </h1>
//                     {images.length>0 && <span className="page-count">{images.length} photos</span>}
//                     {batchMode && selected.size>0 && <span className="sel-count">{selected.size} selected</span>}
//                     {view==="explore" && <button className="btn-sm" style={{marginLeft:"auto"}} onClick={load}><Shuffle size={11}/> Shuffle</button>}
//                   </div>

//                   {images.length===0
//                     ? (view==="search" && peopleSuggestions.length>0 ? (
//                         <div>
//                           <Empty icon={ImageOff} msg="No photos matched" sub="But these people are in your gallery — click to view their photos"/>
//                           <div style={{marginTop:16}}>
//                             <p style={{fontSize:11,color:"var(--mu)",marginBottom:10,fontFamily:"var(--mono)",letterSpacing:".08em"}}>PEOPLE IN YOUR GALLERY:</p>
//                             <div className="people-grid">
//                               {peopleSuggestions.map(p=>(
//                                 <motion.div key={p.id} className="person-card" whileHover={{y:-3}}
//                                   onClick={async()=>{
//                                     setLoading(true); setView("search"); setPeopleSuggestions([]);
//                                     try { const r=await axios.get(`${API}/people/${p.id}`); setImages(r.data.results||r.data.images||[]); }
//                                     catch {} finally { setLoading(false); }
//                                   }}>
//                                   <div className="person-avatar">
//                                     {p.cover?<img src={imgUrl(p.cover)} alt="" onError={e=>e.target.src=BLANK}/>:<Users size={26} strokeWidth={1} color="#444"/>}
//                                   </div>
//                                   <p className="person-name">{p.name}</p>
//                                   <p className="person-count" style={{color:"var(--ac2)",fontSize:9}}>click to rename</p>
//                                 </motion.div>
//                               ))}
//                             </div>
//                             <p style={{fontSize:11,color:"#555",marginTop:14,padding:"10px 14px",background:"var(--s2)",borderRadius:8,border:"1px solid var(--br)"}}>
//                               💡 <strong>Tip:</strong> Click a person above → click ✏ Rename → type their real name → search will work instantly
//                             </p>
//                           </div>
//                         </div>
//                       ) : <Empty icon={ImageOff} msg={view==="search"?"No results":"No photos yet"} sub={view!=="search"?"Upload photos to get started":"Try different keywords"} onClick={view!=="search"?()=>uploadRef.current?.click():null}/>)
//                     : <ImgGrid list={images} onOpen={i=>openLb(images,i)} onFav={toggleFav} onDel={softDelete} batchMode={batchMode} selected={selected} onToggle={toggleSelect}/>
//                   }
//                 </Fade>

//               ) : view==="faces" ? (
//                 <Fade key="faces">
//                   <PageHead title="People" count={faces.length}/>
//                   {faces.length===0
//                     ? <Empty icon={Users} msg="No people detected" sub="Run Re-index to detect faces"/>
//                     : <div className="people-grid">{faces.map(p=><PersonCard key={p.id} p={p} onClick={()=>openPerson(p)}/>)}</div>
//                   }
//                 </Fade>

//               ) : view==="albums" ? (
//                 <Fade key="albums">
//                   <div className="page-head">
//                     <h1 className="page-title">Albums</h1>
//                     {albums.length>0 && <span className="page-count">{albums.length}</span>}
//                     <button className="btn-sm btn-sm--accent" style={{marginLeft:"auto"}} onClick={()=>{ setPendingAlbumImages([]); setCreateAlbumOpen(true); }}>
//                       <Plus size={12}/> New Album
//                     </button>
//                     <button className="btn-sm" style={{marginLeft:6,background:"var(--s3)",border:"1px solid var(--br)"}} onClick={()=>setCreateEventOpen(true)}>🎉 New Event</button>
//                   </div>
//                   {albums.length===0
//                     ? <Empty icon={BookImage} msg="No albums" sub="Run Re-index to auto-generate or create manually"/>
//                     : <div className="albums-grid">{albums.map(a=><AlbumCard key={a.id} a={a} onClick={()=>openAlbum(a)} onDelete={a.type==="manual"?()=>deleteAlbum(a.id):null}/>)}</div>
//                   }
//                 </Fade>

//               ) : view==="duplicates" ? (
//                 <Fade key="dupes">
//                   <PageHead title="Duplicate Groups" count={dupes.length} unit="groups"/>
//                   {dupes.length===0
//                     ? <Empty icon={Copy} msg="No duplicates" sub="All images are unique"/>
//                     : <div className="dupe-list">
//                         {dupes.map((g,gi)=>(
//                           <div key={gi} className="dupe-group">
//                             <div className="dupe-head">
//                               <span className="tag-sm">Group {gi+1}</span>
//                               <span className="tag-sm tag-sm--accent">{g.count} duplicates</span>
//                               {g.total_size&&<span className="tag-sm">{(g.total_size/1024/1024).toFixed(1)} MB</span>}
//                             </div>
//                             <ImgGrid list={g.images||[]} compact onOpen={i=>openLb(g.images,i)} onFav={toggleFav} onDel={softDelete}/>
//                           </div>
//                         ))}
//                       </div>
//                   }
//                 </Fade>

//               ) : view==="trash" ? (
//                 <Fade key="trash">
//                   <PageHead title="Trash" count={trash.length}/>
//                   {trash.length===0 ? <Empty icon={Trash} msg="Trash is empty"/> : (
//                     <div className="img-grid">
//                       {trash.map(img=>(
//                         <div key={img.id} className="trash-card">
//                           <img src={imgUrl(img.filename)} alt="" className="trash-img" onError={e=>e.target.src=BLANK}/>
//                           <div className="trash-overlay">
//                             <button className="trash-btn trash-btn--restore" onClick={()=>restoreImg(img.id)}><RotateCcw size={11}/> Restore</button>
//                             <button className="trash-btn trash-btn--del" onClick={()=>permDelete(img.id)}><AlertTriangle size={11}/> Delete Forever</button>
//                           </div>
//                           {img.trashed_at&&<span className="trash-date">{new Date(img.trashed_at).toLocaleDateString()}</span>}
//                         </div>
//                       ))}
//                     </div>
//                   )}
//                 </Fade>

//               ) : view==="stats" ? (
//                 <Fade key="stats"><StatsPage stats={stats} peopleFreq={peopleFreq}/></Fade>
//               ) : null}
//             </AnimatePresence>
//           </div>

//           {/* BATCH ACTION BAR */}
//           <AnimatePresence>
//             {batchMode && (
//               <motion.div className="batch-bar"
//                 initial={{y:60,opacity:0}} animate={{y:0,opacity:1}} exit={{y:60,opacity:0}}>
//                 <div className="batch-bar-left">
//                   <span className="batch-count"><CheckCheck size={14}/> {selected.size} selected</span>
//                   <button className="batch-sm" onClick={selectAll}>All</button>
//                   <button className="batch-sm" onClick={clearSel}>None</button>
//                 </div>
//                 <div className="batch-bar-actions">
//                   <button className="batch-action-btn batch-action-btn--fav" onClick={()=>batchFavorite(true)} title="Favorite all"><Heart size={13}/> Favorite</button>
//                   <button className="batch-action-btn batch-action-btn--del" onClick={batchDelete} title="Trash all"><Trash2 size={13}/> Trash</button>
//                   <div className="batch-input-group">
//                     <input className="batch-input" placeholder="Add tag…" value={batchTagInput} onChange={e=>setBatchTagInput(e.target.value)} onKeyDown={e=>e.key==="Enter"&&batchTag()}/>
//                     <button className="batch-action-btn" onClick={batchTag}><Tag size={12}/></button>
//                   </div>
//                   <div className="batch-input-group">
//                     <select className="batch-select" value={batchAlbumId} onChange={e=>setBatchAlbumId(e.target.value)}>
//                       <option value="">Add to album…</option>
//                       {albums.map(a=><option key={a.id} value={a.id}>{a.title}</option>)}
//                     </select>
//                     <button className="batch-action-btn" onClick={batchAddAlbum} title="Add to existing album"><Layers size={12}/></button>
//                   </div>
//                   <button className="batch-action-btn batch-action-btn--album"
//                     onClick={()=>{
//                       setPendingAlbumImages([...selected]); // snapshot NOW before modal opens
//                       setCreateAlbumOpen(true);
//                     }}
//                     title={`Create new album with ${selected.size} selected photos`}
//                     disabled={selected.size===0}>
//                     <FolderPlus size={12}/> New Album
//                   </button>
//                 </div>
//               </motion.div>
//             )}
//           </AnimatePresence>
//         </div>
//       </div>

//       {/* LIGHTBOX */}
//       <AnimatePresence>
//         {lb && <Lightbox list={lb.list} idx={lb.idx} onClose={closeLb} onShift={shiftLb} onFav={toggleFav} onDel={(id,e)=>{softDelete(id,e);closeLb();}} onAddTag={addTagToImage} onRemoveTag={removeTagFromImage} albums={albums}/>}
//       </AnimatePresence>

//       {/* PERSON MODAL */}
//       <AnimatePresence>
//         {personModal && <PersonModal data={personModal} onClose={()=>setPersonModal(null)} onRename={renamePerson} onOpen={(list,i)=>{setPersonModal(null);setTimeout(()=>openLb(list,i),80);}}/>}
//       </AnimatePresence>

//       {/* ALBUM MODAL */}
//       <AnimatePresence>
//         {albumModal && <AlbumModal data={albumModal} onClose={()=>setAlbumModal(null)} onRename={renameAlbum} onDelete={deleteAlbum} onOpen={(list,i)=>{setAlbumModal(null);setTimeout(()=>openLb(list,i),80);}}/>}
//       </AnimatePresence>

//       {/* CREATE ALBUM MODAL */}
//       <AnimatePresence>
//         {createAlbumOpen && (
//           <ModalWrap onClose={()=>setCreateAlbumOpen(false)}>
//             <div className="modal-head">
//               <h2 className="modal-title">New Album</h2>
//               <button className="lb-hbtn lb-close" onClick={()=>setCreateAlbumOpen(false)}><X size={17}/></button>
//             </div>
//             {pendingAlbumImages.length>0 && (
//               <div style={{background:"rgba(96,165,250,.08)",border:"1px solid rgba(96,165,250,.2)",borderRadius:8,padding:"8px 12px",marginBottom:12,fontSize:12,color:"rgba(96,165,250,.9)"}}>
//                 📸 <strong>{pendingAlbumImages.length} selected photos</strong> will be added to this album
//               </div>
//             )}
//             <div className="create-album-form">
//               <input className="form-input" placeholder="Album title…" value={newAlbumTitle}
//                 onChange={e=>setNewAlbumTitle(e.target.value)}
//                 onKeyDown={e=>e.key==="Enter"&&newAlbumTitle.trim()&&createAlbum()}
//                 autoFocus/>
//               <textarea className="form-textarea" placeholder="Description (optional)…"
//                 value={newAlbumDesc} onChange={e=>setNewAlbumDesc(e.target.value)} rows={2}/>
//               <div style={{display:"flex",gap:8}}>
//                 <button className="btn-sm btn-sm--primary" style={{flex:1}} onClick={()=>createAlbum()} disabled={!newAlbumTitle.trim()}>
//                   <FolderPlus size={13}/> Create Album
//                 </button>
//                 <button className="btn-sm" onClick={()=>setCreateAlbumOpen(false)}>Cancel</button>
//               </div>
//             </div>
//           </ModalWrap>
//         )}
//       </AnimatePresence>

//       {/* CREATE NAMED EVENT MODAL */}
//       <AnimatePresence>
//         {createEventOpen && (
//           <ModalWrap onClose={()=>setCreateEventOpen(false)}>
//             <div className="modal-head">
//               <h2 className="modal-title">🎉 New Event</h2>
//               <button className="lb-hbtn lb-close" onClick={()=>setCreateEventOpen(false)}><X size={17}/></button>
//             </div>
//             <div className="create-album-form">
//               <label style={{fontSize:11,color:"var(--mu)",marginBottom:4}}>EVENT TYPE</label>
//               <div style={{display:"flex",flexWrap:"wrap",gap:6,marginBottom:12}}>
//                 {["Birthday","Trip","Vacation","Wedding","Anniversary","Graduation","Party","Holiday","Family","Work","Other"].map(t=>(
//                   <button key={t}
//                     className={"btn-sm" + (newEventType===t?" btn-sm--primary":"")}
//                     style={{fontSize:11,padding:"4px 10px"}}
//                     onClick={()=>setNewEventType(t)}>
//                     {t==="Birthday"?"🎂":t==="Trip"?"✈️":t==="Vacation"?"🏖️":t==="Wedding"?"💍":t==="Anniversary"?"💕":t==="Graduation"?"🎓":t==="Party"?"🎊":t==="Holiday"?"🎄":t==="Family"?"👨‍👩‍👧":t==="Work"?"💼":"📅"} {t}
//                   </button>
//                 ))}
//               </div>
//               <input className="form-input" placeholder="Event name (e.g. Mom's 60th Birthday)…" value={newEventTitle} onChange={e=>setNewEventTitle(e.target.value)} autoFocus/>
//               <input className="form-input" type="date" value={newEventDate} onChange={e=>setNewEventDate(e.target.value)} style={{marginTop:8}}/>
//               <textarea className="form-textarea" placeholder="Notes (optional)…" value={newEventDesc} onChange={e=>setNewEventDesc(e.target.value)} rows={2} style={{marginTop:8}}/>
//               <button className="btn-sm btn-sm--primary" onClick={createNamedEvent} disabled={!newEventTitle.trim()} style={{marginTop:4}}>
//                 🎉 Create Event
//               </button>
//             </div>
//           </ModalWrap>
//         )}
//       </AnimatePresence>

//       {/* CO-OCCURRENCE MODAL */}
//       <AnimatePresence>
//         {showCoModal && (
//           <ModalWrap onClose={()=>setShowCoModal(false)}>
//             <div className="modal-head">
//               <h2 className="modal-title"><GitMerge size={16} style={{marginRight:6,verticalAlign:"middle"}}/>Co-Appearance Search</h2>
//               <button className="lb-hbtn lb-close" onClick={()=>setShowCoModal(false)}><X size={17}/></button>
//             </div>
//             <CoOccurrencePanel faces={faces} onSearch={runCoOccurrence}/>
//           </ModalWrap>
//         )}
//       </AnimatePresence>

//       {/* FACE SIMILARITY MODAL */}
//       <AnimatePresence>
//         {showFaceSim && (
//           <ModalWrap onClose={()=>setShowFaceSim(false)}>
//             <div className="modal-head">
//               <h2 className="modal-title"><UserCheck size={16} style={{marginRight:6,verticalAlign:"middle"}}/>Face Similarity Search</h2>
//               <button className="lb-hbtn lb-close" onClick={()=>setShowFaceSim(false)}><X size={17}/></button>
//             </div>
//             <div className="create-album-form">
//               <p style={{fontSize:12,color:"var(--mu)",marginBottom:12}}>Upload a face crop or photo — find all photos with the same person.</p>
//               {faceSrcPrev && <img src={faceSrcPrev} style={{width:120,height:120,objectFit:"cover",borderRadius:8,marginBottom:12,border:"2px solid var(--ac)"}} alt="face"/>}
//               <input type="file" accept="image/*" style={{display:"none"}} id="face-sim-input"
//                 onChange={e=>{
//                   const f=e.target.files[0]; if(!f) return;
//                   setFaceSrcFile(f);
//                   const r=new FileReader(); r.onload=ev=>setFaceSrcPrev(ev.target.result); r.readAsDataURL(f);
//                 }}/>
//               <button className="btn-sm" onClick={()=>document.getElementById("face-sim-input").click()}>
//                 <Camera size={12}/> {faceSrcPrev?"Change Photo":"Upload Face Photo"}
//               </button>
//               {faceSrcFile && (
//                 <button className="btn-sm btn-sm--primary" style={{marginTop:10}} onClick={runFaceSimilarity}>
//                   <UserCheck size={12}/> Find Similar Faces
//                 </button>
//               )}
//             </div>
//           </ModalWrap>
//         )}
//       </AnimatePresence>
//     </>
//   );
// }

// /* ══════════════════════════════════════════════════════════════════════════ */
// /* ON THIS DAY BANNER                                                          */
// /* ══════════════════════════════════════════════════════════════════════════ */
// function OnThisDayBanner({ data, expanded, onToggle, onOpen }) {
//   return (
//     <div className="otd-banner">
//       <div className="otd-header" onClick={onToggle}>
//         <div className="otd-left">
//           <Calendar size={15} color="#f59e0b"/>
//           <div>
//             <span className="otd-title">On This Day — {data.date}</span>
//             <span className="otd-sub">{data.total} photo{data.total!==1?"s":""} from {data.years?.length} year{data.years?.length!==1?"s":""} ago</span>
//           </div>
//         </div>
//         <button className="otd-toggle">{expanded?<ChevronUp size={14}/>:<ChevronDown size={14}/>}</button>
//       </div>
//       <AnimatePresence>
//         {expanded && (
//           <motion.div initial={{height:0,opacity:0}} animate={{height:"auto",opacity:1}} exit={{height:0,opacity:0}}>
//             {data.years?.map(yr=>(
//               <div key={yr.year} className="otd-year">
//                 <p className="otd-year-label">{yr.year} — {yr.count} photo{yr.count!==1?"s":""}</p>
//                 <div className="otd-strip">
//                   {yr.images.map((img,i)=>(
//                     <div key={img.id} className="otd-thumb" onClick={()=>onOpen(yr.images,i)}>
//                       <img src={imgUrl(img.filename)} alt="" onError={e=>e.target.src=BLANK}/>
//                       {img.caption_short && <div className="otd-thumb-caption">{img.caption_short}</div>}
//                     </div>
//                   ))}
//                 </div>
//               </div>
//             ))}
//           </motion.div>
//         )}
//       </AnimatePresence>
//     </div>
//   );
// }


// /* ══════════════════════════════════════════════════════════════════════════ */
// /* IMAGE GRID                                                                  */
// /* ══════════════════════════════════════════════════════════════════════════ */
// function ImgGrid({ list, onOpen, onFav, onDel, compact, batchMode, selected, onToggle }) {
//   return (
//     <div className={`img-grid${compact?" img-grid--sm":""}`}>
//       {list.map((img,i)=>(
//         <PhotoCard key={img.id??i} img={img} onOpen={()=>onOpen(i)} onFav={onFav} onDel={onDel}
//           batchMode={batchMode} isSelected={selected?.has(img.id)} onToggle={onToggle}/>
//       ))}
//     </div>
//   );
// }

// function PhotoCard({ img, onOpen, onFav, onDel, batchMode, isSelected, onToggle }) {
//   const q=qInfo(img.quality_level); const emo=EMO[img.dominant_emotion]||"";
//   return (
//     <motion.div
//       className={`photo-card ${isSelected?"photo-card--sel":""}`}
//       whileHover={{y:-4, scale:1.02}}
//       whileTap={{scale:0.97}}
//       transition={{type:"spring", stiffness:380, damping:28}}
//       onClick={batchMode?()=>onToggle(img.id):onOpen}
//     >
//       <img src={imgUrl(img.filename)||BLANK} alt="" className="photo-img" loading="lazy"
//         onError={e=>{e.target.onerror=null;e.target.src=BLANK;}}/>

//       {/* Top chips */}
//       {img.quality_level&&img.quality_level!=="Processing..."&&(
//         <span className="chip chip--q" style={{color:q.c,background:q.bg,borderColor:q.c+"44"}}>{img.quality_level[0]}</span>
//       )}
//       {emo&&img.dominant_emotion!=="neutral"&&<span className="chip chip--emo">{emo}</span>}
//       {img.is_favorite&&<span className="fav-dot">♥</span>}

//       {batchMode ? (
//         <motion.div className="batch-check" initial={{scale:0.5}} animate={{scale:1}}>
//           {isSelected
//             ? <CheckSquare size={20} color="#7c6af7" style={{filter:"drop-shadow(0 0 6px #7c6af7aa)"}}/>
//             : <Square size={20} color="rgba(255,255,255,.4)"/>
//           }
//         </motion.div>
//       ) : (
//         <div className="photo-hover">
//           <div className="photo-actions">
//             <motion.button whileHover={{scale:1.15}} whileTap={{scale:.9}}
//               className="ph-btn ph-btn--fav" onClick={e=>{e.stopPropagation();onFav(img.id,e);}}>
//               <Heart size={12} fill={img.is_favorite?"#fff":"none"}/>
//             </motion.button>
//             <motion.button whileHover={{scale:1.15}} whileTap={{scale:.9}}
//               className="ph-btn ph-btn--del" onClick={e=>{e.stopPropagation();onDel(img.id,e);}}>
//               <Trash2 size={12}/>
//             </motion.button>
//           </div>
//           <div className="photo-info">
//             {img.caption_short&&<p className="photo-caption">{img.caption_short}</p>}
//             <div className="photo-meta-row">
//               {img.quality_level&&img.quality_level!=="Processing..."&&
//                 <span className="meta-chip" style={{color:q.c}}>◆ {img.quality_level}</span>}
//               {img.aesthetic_score>0&&
//                 <span className="meta-chip" style={{color:"#fbbf24"}}>★ {Number(img.aesthetic_score).toFixed(1)}</span>}
//               {img.dominant_emotion&&img.dominant_emotion!=="neutral"&&
//                 <span className="meta-chip">{emo} {img.dominant_emotion}</span>}
//               {img.user_tags?.length>0&&
//                 <span className="meta-chip" style={{color:"#86efac"}}>#{img.user_tags[0]}{img.user_tags.length>1?` +${img.user_tags.length-1}`:""}</span>}
//             </div>
//           </div>
//         </div>
//       )}
//     </motion.div>
//   );
// }

// /* ══════════════════════════════════════════════════════════════════════════ */
// /* LIGHTBOX                                                                    */
// /* ══════════════════════════════════════════════════════════════════════════ */
// function Lightbox({ list, idx, onClose, onShift, onFav, onDel, onAddTag, onRemoveTag, albums }) {
//   const img=list[idx]; if(!img) return null;
//   const q=qInfo(img.quality_level); const emo=EMO[img.dominant_emotion]||"😐";
//   const [tagInput, setTagInput] = useState("");
//   const [imgTags,  setImgTags]  = useState(img.user_tags||[]);
//   const [addAlbum, setAddAlbum] = useState("");
//   const [note,     setNote]     = useState(img.photo_note||"");
//   const [noteEditing, setNoteEditing] = useState(false);
//   const [noteSaving,  setNoteSaving]  = useState(false);

//   // Always fetch latest tags + note from server when photo changes
//   useEffect(()=>{
//     setTagInput(""); setNoteEditing(false);
//     setImgTags(img.user_tags||[]);
//     setNote(img.photo_note||"");
//     axios.get(`${API}/photo/${img.id}/tags`)
//       .then(r => setImgTags(r.data.tags||[]))
//       .catch(()=>{});
//     axios.get(`${API}/photo/${img.id}/note`)
//       .then(r => setNote(r.data.note||""))
//       .catch(()=>{});
//   }, [img.id]);

//   const saveNote = async () => {
//     setNoteSaving(true);
//     const fd=new FormData(); fd.append("note", note);
//     try { await axios.post(`${API}/photo/${img.id}/note`, fd); setNoteEditing(false); }
//     catch(e) { alert("Failed to save note"); }
//     setNoteSaving(false);
//   };

//   const doAddTag = async () => {
//     if (!tagInput.trim()) return;
//     const newTags = await onAddTag(img.id, tagInput.trim());
//     if (newTags) setImgTags(newTags);
//     else setImgTags(p=>[...new Set([...p, tagInput.trim().toLowerCase()])]);
//     setTagInput("");
//   };
//   const doRemoveTag = async tag => {
//     const newTags = await onRemoveTag(img.id, tag);
//     if (newTags) setImgTags(newTags);
//     else setImgTags(p=>p.filter(t=>t!==tag));
//   };
//   const doAddToAlbum = async () => {
//     if (!addAlbum) return;
//     const fd=new FormData(); fd.append("image_ids",img.id);
//     try { await axios.post(`${API}/batch/album`,fd.set?fd:(()=>{fd.append("album_id",addAlbum);return fd;})()); } catch {}
//     setAddAlbum("");
//   };

//   return (
//     <motion.div className="lb-bg" initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}} onClick={onClose}>
//       <motion.div className="lb-box" initial={{scale:.96}} animate={{scale:1}} exit={{scale:.96}} onClick={e=>e.stopPropagation()}>
//         <div className="lb-left">
//           {idx>0&&<button className="lb-nav lb-nav--l" onClick={()=>onShift(-1)}><ChevronLeft size={20}/></button>}
//           <AnimatePresence mode="wait">
//             <motion.img key={img.id??idx} src={imgUrl(img.filename)||BLANK} alt="" className="lb-img"
//               initial={{opacity:0,x:16}} animate={{opacity:1,x:0}} exit={{opacity:0,x:-16}} transition={{duration:.18}}
//               onError={e=>{e.target.onerror=null;e.target.src=BLANK;}}/>
//           </AnimatePresence>
//           {idx<list.length-1&&<button className="lb-nav lb-nav--r" onClick={()=>onShift(1)}><ChevronRight size={20}/></button>}
//           <div className="lb-counter">{idx+1} / {list.length}</div>
//           <div className="lb-hint">← → · ESC</div>
//         </div>

//         <div className="lb-right">
//           <div className="lb-right-head">
//             <h3 className="lb-right-title">Details</h3>
//             <div className="lb-head-btns">
//               <button className="lb-hbtn" onClick={()=>onFav(img.id)} style={{color:img.is_favorite?"#f43f5e":undefined}}>
//                 <Heart size={15} fill={img.is_favorite?"#f43f5e":"none"} stroke={img.is_favorite?"#f43f5e":"currentColor"}/>
//               </button>
//               <button className="lb-hbtn" onClick={e=>onDel(img.id,e)}><Trash2 size={15}/></button>
//               <button className="lb-hbtn lb-close" onClick={onClose}><X size={17}/></button>
//             </div>
//           </div>

//           <div className="lb-scroll">
//             {img.caption_short&&(
//               <DB icon={<Camera size={12}/>} label="AI Caption" c="#818cf8">
//                 <p className="db-caption">{img.caption_short}</p>
//                 {img.caption_detailed&&img.caption_detailed!==img.caption_short&&<p className="db-caption-sub">{img.caption_detailed}</p>}
//               </DB>
//             )}
//             {img.quality_score>0&&(
//               <DB icon={<TrendingUp size={12}/>} label="Quality" c={q.c}>
//                 <div className="q-wrap">
//                   <div className="q-circle" style={{borderColor:q.c}}>
//                     <span style={{color:q.c,fontSize:18,fontWeight:800}}>{Math.round(img.quality_score)}</span>
//                   </div>
//                   <div className="q-right">
//                     <p className="q-level" style={{color:q.c}}>{img.quality_level}</p>
//                     {img.sharpness>0&&<><QBar label="Sharpness" val={img.sharpness} c={q.c}/><QBar label="Exposure" val={img.exposure} c={q.c}/><QBar label="Contrast" val={img.contrast} c={q.c}/><QBar label="Composition" val={img.composition} c={q.c}/></>}
//                   </div>
//                 </div>
//               </DB>
//             )}
//             {img.aesthetic_score>0&&(
//               <DB icon={<Star size={12}/>} label="Aesthetic" c="#fbbf24">
//                 <div className="aes-row"><span className="aes-score">★ {Number(img.aesthetic_score).toFixed(1)}</span>{img.aesthetic_rating&&<span className="aes-rating">{img.aesthetic_rating}</span>}</div>
//               </DB>
//             )}
//             {img.dominant_emotion&&img.dominant_emotion!=="neutral"&&(
//               <DB icon={<Smile size={12}/>} label="Emotion" c={EMO_COLORS[img.dominant_emotion]||"#f9a8d4"}>
//                 <div className="emo-row">
//                   <span style={{fontSize:30}}>{EMO[img.dominant_emotion]}</span>
//                   <div><p className="emo-name">{img.dominant_emotion}</p>{img.face_emotion_count>0&&<p className="emo-sub">{img.face_emotion_count} face{img.face_emotion_count>1?"s":""}</p>}</div>
//                 </div>
//               </DB>
//             )}
//             {img.ocr_text_enhanced?.trim()&&(
//               <DB icon={<Type size={12}/>} label="Text in Photo" c="#22d3ee">
//                 <p className="ocr-text">{img.ocr_text_enhanced}</p>
//               </DB>
//             )}

//             {/* Tags block */}
//             <DB icon={<FileText size={12}/>} label="Personal Note" c="#fcd34d">
//               {noteEditing ? (
//                 <div>
//                   <textarea className="form-textarea" value={note} onChange={e=>setNote(e.target.value)}
//                     placeholder="Write a personal note about this photo…" rows={3} autoFocus
//                     style={{marginBottom:6,fontSize:12}}/>
//                   <div style={{display:"flex",gap:6}}>
//                     <button className="btn-sm btn-sm--primary" onClick={saveNote} disabled={noteSaving}>
//                       {noteSaving?"Saving…":"Save Note"}
//                     </button>
//                     <button className="btn-sm" onClick={()=>setNoteEditing(false)}>Cancel</button>
//                   </div>
//                 </div>
//               ) : (
//                 <div>
//                   {note ? <p style={{fontSize:12,color:"var(--tx)",lineHeight:1.5,marginBottom:6}}>{note}</p>
//                         : <p style={{fontSize:11,color:"var(--mu)"}}>No note yet</p>}
//                   <button className="btn-sm" style={{fontSize:10,padding:"3px 8px"}} onClick={()=>setNoteEditing(true)}>
//                     ✏ {note?"Edit Note":"Add Note"}
//                   </button>
//                 </div>
//               )}
//             </DB>

//             <DB icon={<Hash size={12}/>} label="Tags" c="#86efac">
//               <div className="tag-chips">
//                 {imgTags.map(t=>(
//                   <span key={t} className="tag-chip-item">
//                     #{t}
//                     <button className="tag-chip-x" onClick={()=>doRemoveTag(t)}><X size={8}/></button>
//                   </span>
//                 ))}
//                 {imgTags.length===0&&<span style={{fontSize:10,color:"var(--mu)"}}>No tags yet</span>}
//               </div>
//               <div className="tag-add-row">
//                 <input className="tag-add-input" placeholder="Add tag…" value={tagInput} onChange={e=>setTagInput(e.target.value)} onKeyDown={e=>e.key==="Enter"&&doAddTag()}/>
//                 <button className="tag-add-btn" onClick={doAddTag}><Plus size={12}/></button>
//               </div>
//             </DB>

//             {/* Add to album */}
//             {albums?.length>0&&(
//               <DB icon={<Layers size={12}/>} label="Add to Album" c="#a78bfa">
//                 <div className="tag-add-row">
//                   <select className="batch-select" style={{flex:1}} value={addAlbum} onChange={e=>setAddAlbum(e.target.value)}>
//                     <option value="">Choose album…</option>
//                     {albums.map(a=><option key={a.id} value={a.id}>{a.title}</option>)}
//                   </select>
//                   <button className="tag-add-btn" onClick={doAddToAlbum}><Plus size={12}/></button>
//                 </div>
//               </DB>
//             )}

//             {img.scene_label&&(
//               <DB icon={<Tag size={12}/>} label="Objects" c="#86efac">
//                 <div className="tag-wrap">{img.scene_label.split(",").map((t,i)=><span key={i} className="obj-tag">{t.trim()}</span>)}</div>
//               </DB>
//             )}
//             <DB icon={<Info size={12}/>} label="Metadata" c="#6b7280">
//               <div className="meta-stack">
//                 {img.timestamp&&<MRow k="Captured" v={new Date(img.timestamp).toLocaleString()}/>}
//                 {img.width&&<MRow k="Dimensions" v={`${img.width} × ${img.height}`}/>}
//                 {img.person_count>0&&<MRow k="People" v={`${img.person_count} detected`}/>}
//                 {img.score>0&&<MRow k="Relevance" v={`${Math.round(img.score)}%`}/>}
//               </div>
//             </DB>
//           </div>
//         </div>
//       </motion.div>
//     </motion.div>
//   );
// }

// function QBar({label,val,c}){ return <div className="qbar-row"><span className="qbar-label">{label}</span><div className="qbar-track"><div className="qbar-fill" style={{width:`${Math.min(100,(val||0)*100)}%`,background:c}}/></div><span className="qbar-num">{Math.round((val||0)*100)}</span></div>; }
// function MRow({k,v}){ return <div className="mrow"><span className="mrow-k">{k}</span><span className="mrow-v">{v}</span></div>; }
// function DB({icon,label,c,children}){ return <div className="db" style={{"--dc":c}}><div className="db-head">{React.cloneElement(icon,{color:c})}<span className="db-label">{label}</span></div><div className="db-body">{children}</div></div>; }

// /* Person / Album components */
// function PersonCard({p,onClick}){ return <motion.div className="person-card" whileHover={{y:-3}} onClick={onClick}><div className="person-avatar">{p.cover?<img src={imgUrl(p.cover)} alt="" onError={e=>e.target.src=BLANK}/>:<Users size={26} strokeWidth={1} color="#444"/>}</div><p className="person-name">{p.name}</p><p className="person-count">{p.count||p.face_count||0} photos</p></motion.div>; }

// function PersonModal({data,onClose,onRename,onOpen}){
//   const [editing, setEditing] = useState(false);
//   const [name, setName] = useState(data.name || "");
//   const imgs = data.images || [];
//   const isDefault = /^Person \d+$/.test(data.name || "");

//   const doSave = () => {
//     if (name.trim()) { onRename(data.id, name.trim()); setEditing(false); }
//   };

//   return (
//     <ModalWrap onClose={onClose} wide>
//       <div className="modal-head">
//         <div>
//           <h2 className="modal-title">{data.name}</h2>
//           <p className="modal-sub">{data.face_count||data.count||0} photos</p>
//         </div>
//         <div style={{display:"flex",gap:6,alignItems:"center"}}>
//           <button className="btn-sm btn-sm--primary" onClick={()=>{setEditing(true);setName(isDefault?"":data.name);}}>
//             ✏ Rename
//           </button>
//           <button className="lb-hbtn lb-close" onClick={onClose}><X size={17}/></button>
//         </div>
//       </div>

//       {isDefault && !editing && (
//         <div style={{background:"rgba(250,204,21,.07)",border:"1px solid rgba(250,204,21,.2)",borderRadius:8,padding:"10px 14px",marginBottom:14}}>
//           <p style={{fontSize:12,color:"rgba(250,204,21,.9)"}}>
//             💡 <strong>Give this person a name</strong> so you can search for their photos. Click <strong>✏ Rename</strong> above.
//           </p>
//         </div>
//       )}

//       {editing && (
//         <div style={{background:"var(--s2)",border:"1px solid var(--br)",borderRadius:10,padding:"14px 16px",marginBottom:14}}>
//           <p style={{fontSize:11,color:"var(--mu)",marginBottom:8,fontFamily:"var(--mono)",letterSpacing:".07em"}}>
//             ENTER PERSON'S NAME
//           </p>
//           <div style={{display:"flex",gap:8,alignItems:"center"}}>
//             <input
//               className="rename-in"
//               style={{flex:1,fontSize:15,padding:"10px 14px"}}
//               value={name}
//               onChange={e=>setName(e.target.value)}
//               onKeyDown={e=>{if(e.key==="Enter")doSave(); if(e.key==="Escape")setEditing(false);}}
//               placeholder="e.g. Vijay, Shah Rukh Khan…"
//               autoFocus
//             />
//             <button className="btn-sm btn-sm--primary" style={{padding:"10px 18px",fontSize:13}} onClick={doSave} disabled={!name.trim()}>
//               Save
//             </button>
//             <button className="btn-sm" onClick={()=>setEditing(false)}>Cancel</button>
//           </div>
//           <p style={{fontSize:10,color:"#555",marginTop:8}}>
//             After saving, you can search for this name in the search bar to find all their photos.
//           </p>
//         </div>
//       )}

//       {data.images===null
//         ? <div className="loader-wrap" style={{minHeight:140}}><div className="loader-ring"/></div>
//         : imgs.length>0
//           ? <ImgGrid list={imgs} onOpen={i=>onOpen(imgs,i)} onFav={()=>{}} onDel={()=>{}}/>
//           : <p className="empty-sub">No photos for this person</p>
//       }
//     </ModalWrap>
//   );
// }

// function AlbumCard({a,onClick,onDelete}){
//   return (
//     <motion.div className="album-card" whileHover={{y:-3,scale:1.01}} onClick={onClick}>
//       {a.cover
//         ? <img src={imgUrl(a.cover)} alt="" className="album-cover" onError={e=>e.target.src=BLANK}/>
//         : <div className="album-cover-ph"><FolderPlus size={32} strokeWidth={1} color="#555"/><span style={{fontSize:10,color:"#555",marginTop:4}}>Empty</span></div>
//       }
//       {a.thumbnails?.length>1&&<div className="album-strip">{a.thumbnails.slice(1,5).map((fn,i)=><img key={i} src={imgUrl(fn)} alt="" className="album-strip-img" onError={e=>e.target.style.display="none"}/>)}</div>}
//       <div className="album-info">
//         <h3 className="album-title">{a.title}</h3>
//         <div className="album-meta">
//           {(()=>{
//   const typeEmoji = {Birthday:"🎂",Trip:"✈️",Vacation:"🏖️",Wedding:"💍",Anniversary:"💕",Graduation:"🎓",Party:"🎊",Holiday:"🎄",Family:"👨‍👩‍👧",Work:"💼"};
//   const desc = a.description||"";
//   const etype = Object.keys(typeEmoji).find(k=>desc.startsWith(k));
//   const label = a.type==="manual"?"✎ album": etype ? typeEmoji[etype]+" "+etype : a.type||"event";
//   const cls = a.type==="manual"?"tag-sm--manual":"tag-sm--accent";
//   return <span className={`tag-sm ${cls}`}>{label}</span>;
// })()}
//           {a.date&&<span className="tag-sm">{a.date}</span>}
//           <span className="tag-sm">📸 {a.count}</span>
//         </div>
//       </div>
//       {onDelete&&<button className="album-del-btn" onClick={e=>{e.stopPropagation();onDelete();}} title="Delete album"><X size={12}/></button>}
//     </motion.div>
//   );
// }

// function AlbumModal({data,onClose,onOpen,onRename,onDelete}){
//   const imgs=data.images||[];
//   const [editing,setEditing]=React.useState(false);
//   const [title,setTitle]=React.useState(data.title||"");
//   const [desc,setDesc]=React.useState(data.description||"");
//   const saveRename=()=>{ onRename&&onRename(data.id,title,desc); setEditing(false); };
//   return (
//     <ModalWrap onClose={onClose} wide>
//       <div className="modal-head">
//         <div>
//           <h2 className="modal-title">{data.title}</h2>
//           <p className="modal-sub">{data.image_count||data.count||0} photos{data.date?` · ${data.date}`:""}</p>
//         </div>
//         <div style={{display:"flex",gap:6,alignItems:"center"}}>
//           <button className="btn-sm" onClick={()=>setEditing(e=>!e)}>✏ Rename</button>
//           {data.type==="manual"&&onDelete&&(
//             <button className="btn-sm" style={{color:"#f87171",borderColor:"rgba(248,113,113,.3)"}}
//               onClick={()=>{ if(window.confirm(`Delete album "${data.title}"? Photos are kept.`)){onDelete(data.id);onClose();} }}>
//               🗑 Delete
//             </button>
//           )}
//           <button className="lb-hbtn lb-close" onClick={onClose}><X size={17}/></button>
//         </div>
//       </div>
//       {editing&&(
//         <div className="rename-row" style={{marginBottom:12}}>
//           <input className="rename-in" value={title} onChange={e=>setTitle(e.target.value)}
//             onKeyDown={e=>e.key==="Enter"&&saveRename()} placeholder="Album name…" autoFocus/>
//           <input className="rename-in" value={desc} onChange={e=>setDesc(e.target.value)}
//             placeholder="Description (optional)…" style={{marginLeft:6}}/>
//           <button className="btn-sm btn-sm--primary" onClick={saveRename} style={{marginLeft:6}}>Save</button>
//           <button className="btn-sm" onClick={()=>setEditing(false)} style={{marginLeft:4}}>Cancel</button>
//         </div>
//       )}
//       {!editing&&data.description&&(
//         <div className="album-desc"><span className="album-desc-lbl">DESCRIPTION</span><p>{data.description}</p></div>
//       )}
//       {data.images===null
//         ? <div className="loader-wrap" style={{minHeight:140}}><div className="loader-ring"/></div>
//         : imgs.length>0
//           ? <ImgGrid list={imgs} onOpen={i=>onOpen(imgs,i)} onFav={()=>{}} onDel={()=>{}}/>
//           : (
//             <div style={{textAlign:"center",padding:"40px 20px",color:"var(--mu)"}}>
//               <FolderPlus size={40} strokeWidth={1} style={{marginBottom:12,opacity:.4}}/>
//               <p style={{fontSize:14,marginBottom:6}}>This album is empty</p>
//               <p style={{fontSize:12,color:"#444"}}>
//                 Go to <strong>Timeline</strong> → select photos (checkbox icon) →
//                 choose this album in the batch bar → click the layers icon
//               </p>
//             </div>
//           )
//       }
//     </ModalWrap>
//   );
// }

// function ModalWrap({children,onClose,wide}){
//   return <motion.div className="modal-bg" initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}} onClick={onClose}><motion.div className="modal-box" style={{maxWidth:wide?960:680}} initial={{scale:.95,y:12}} animate={{scale:1,y:0}} exit={{scale:.95}} onClick={e=>e.stopPropagation()}>{children}</motion.div></motion.div>;
// }

// function CoOccurrencePanel({faces, onSearch}) {
//   const [selected, setSelected] = React.useState([]);
//   const toggle = id => setSelected(p => p.includes(id) ? p.filter(x=>x!==id) : [...p, id]);
//   return (
//     <div className="create-album-form">
//       <p style={{fontSize:12,color:"var(--mu)",marginBottom:10}}>Select 2+ people to find photos where they all appear together:</p>
//       <div style={{display:"flex",flexWrap:"wrap",gap:8,marginBottom:14,maxHeight:240,overflowY:"auto"}}>
//         {(faces||[]).map(p=>(
//           <button key={p.id}
//             className={"person-chip" + (selected.includes(p.id)?" person-chip--on":"")}
//             onClick={()=>toggle(p.id)}>
//             {p.cover && <img src={imgUrl(p.cover)} alt="" style={{width:24,height:24,borderRadius:"50%",objectFit:"cover",marginRight:6}}/>}
//             {p.name||`Person ${p.id}`}
//           </button>
//         ))}
//         {(!faces||faces.length===0) && <p style={{fontSize:12,color:"var(--mu)"}}>No named people yet. Go to People section first.</p>}
//       </div>
//       <button className="btn-sm btn-sm--primary" disabled={selected.length<2} onClick={()=>onSearch(selected)}>
//         <GitMerge size={12}/> Find Co-appearing Photos ({selected.length} selected)
//       </button>
//     </div>
//   );
// }

// function EmotionTimelineChart() {
//   const [data, setData] = React.useState(null);
//   React.useEffect(()=>{
//     axios.get(`${API}/emotion-timeline`).then(r=>setData(r.data)).catch(()=>{});
//   },[]);

//   if (!data) return <p style={{fontSize:12,color:"var(--mu)",padding:8}}>Loading…</p>;
//   if (!data.months?.length) return (
//     <div style={{padding:"12px 0"}}>
//       <p style={{fontSize:12,color:"var(--mu)"}}>No emotion timeline data yet.</p>
//       <p style={{fontSize:11,color:"#555",marginTop:4}}>
//         Emotions are detected in the background after uploading. Install <code style={{background:"var(--s3)",padding:"1px 5px",borderRadius:3}}>pip install fer</code> for best results, then click <strong>Re-index AI</strong>.
//       </p>
//     </div>
//   );

//   // Show server hint if only neutral detected
//   const serverHint = data.hint;

//   const EMO_COL = {happy:"#4ade80",sad:"#60a5fa",angry:"#f87171",neutral:"#a8a29e",surprised:"#fbbf24",disgusted:"#a78bfa",fearful:"#f9a8d4"};
//   const EMO_EMOJI = {happy:"😊",sad:"😢",angry:"😠",neutral:"😐",surprised:"😲",disgusted:"🤢",fearful:"😨"};

//   const months = data.months;
//   const n = months.length;
//   // Filter to emotions that actually have data (exclude neutral-only if others exist)
//   const showEmos = data.emotions.filter(e => (data.series[e]||[]).some(v=>v>0));
//   const hasRealData = showEmos.some(e => e !== "neutral");

//   // Chart dimensions — extra bottom padding for legend
//   const W=560, CHART_H=160, LEG_H=28, PAD={l:36,r:16,t:12,b:32};
//   const chartW = W - PAD.l - PAD.r;
//   const chartH = CHART_H - PAD.t - PAD.b;
//   const maxVal = Math.max(1, ...showEmos.flatMap(e => data.series[e]||[]));

//   const xScale = i => PAD.l + (n <= 1 ? chartW/2 : (i / (n-1)) * chartW);
//   const yScale = v => PAD.t + chartH - (v / maxVal) * chartH;
//   const makePath = series => series.map((v,i) =>
//     `${i===0?"M":"L"}${xScale(i).toFixed(1)},${yScale(v).toFixed(1)}`
//   ).join(" ");

//   // X labels: show at most 8, evenly spaced
//   const maxLabels = Math.min(8, n);
//   const labelIndices = n <= maxLabels
//     ? months.map((_,i)=>i)
//     : Array.from({length:maxLabels}, (_,k) => Math.round(k*(n-1)/(maxLabels-1)));

//   return (
//     <div style={{overflowX:"auto"}}>
//       {(serverHint || !hasRealData) && (
//         <p style={{fontSize:11,color:"#f59e0b",marginBottom:8,background:"rgba(245,158,11,.08)",padding:"6px 10px",borderRadius:6,border:"1px solid rgba(245,158,11,.2)"}}>
//           ⚠️ {serverHint || 'Only "neutral" detected — run pip install fer then Re-index AI.'}
//         </p>
//       )}
//       <svg viewBox={`0 0 ${W} ${CHART_H + LEG_H}`} style={{width:"100%",maxWidth:W,display:"block"}}>
//         {/* Y grid + labels */}
//         {[0,0.25,0.5,0.75,1].map((f,i)=>{
//           const y = PAD.t + chartH*(1-f);
//           return (
//             <g key={i}>
//               <line x1={PAD.l} x2={W-PAD.r} y1={y} y2={y} stroke="#ffffff08" strokeWidth={1}/>
//               <text x={PAD.l-4} y={y+3} fontSize={7} fill="#555" textAnchor="end">{Math.round(maxVal*f)}</text>
//             </g>
//           );
//         })}
//         {/* Axes */}
//         <line x1={PAD.l} x2={PAD.l} y1={PAD.t} y2={PAD.t+chartH} stroke="#333" strokeWidth={1}/>
//         <line x1={PAD.l} x2={W-PAD.r} y1={PAD.t+chartH} y2={PAD.t+chartH} stroke="#333" strokeWidth={1}/>
//         {/* Emotion lines + dots */}
//         {showEmos.map(e=>(
//           <g key={e}>
//             <path d={makePath(data.series[e]||[])} fill="none" stroke={EMO_COL[e]||"#888"} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round"/>
//             {(data.series[e]||[]).map((v,i)=> v>0 && (
//               <circle key={i} cx={xScale(i)} cy={yScale(v)} r={3} fill={EMO_COL[e]||"#888"} opacity={0.85}/>
//             ))}
//           </g>
//         ))}
//         {/* X axis labels */}
//         {labelIndices.map(i=>(
//           <text key={i} x={xScale(i)} y={PAD.t+chartH+14} fontSize={7} fill="#555" textAnchor="middle">
//             {months[i]}
//           </text>
//         ))}
//         {/* Legend — below chart, wrapping row */}
//         {showEmos.map((e,i)=>{
//           const cols = Math.min(showEmos.length, 4);
//           const lx = PAD.l + (i % cols) * 130;
//           const ly = CHART_H + Math.floor(i/cols)*14 + 4;
//           return (
//             <g key={e} transform={`translate(${lx},${ly})`}>
//               <rect width={8} height={8} fill={EMO_COL[e]||"#888"} rx={2}/>
//               <text x={11} y={7} fontSize={9} fill="#aaa">{EMO_EMOJI[e]} {e}</text>
//             </g>
//           );
//         })}
//       </svg>
//     </div>
//   );
// }

// function PeopleFreqChart({people}) {
//   if(!people?.length) return <p style={{fontSize:12,color:"var(--mu)",padding:8}}>No people data yet.</p>;
//   const top = people.slice(0,12);
//   const maxN = Math.max(...top.map(p=>p.count),1);
//   return (
//     <div style={{display:"flex",flexDirection:"column",gap:6}}>
//       {top.map((p,i)=>(
//         <div key={p.id} style={{display:"flex",alignItems:"center",gap:8}}>
//           <span style={{fontSize:11,color:"var(--mu)",width:20,textAlign:"right"}}>{i+1}</span>
//           <span style={{fontSize:12,color:"var(--tx)",width:120,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{p.name||`Person ${p.id}`}</span>
//           <div style={{flex:1,height:14,background:"var(--s3)",borderRadius:4,overflow:"hidden"}}>
//             <div style={{width:`${(p.count/maxN)*100}%`,height:"100%",background:"linear-gradient(90deg,#818cf8,#a78bfa)",borderRadius:4,transition:"width .4s"}}/>
//           </div>
//           <span style={{fontSize:11,color:"#818cf8",width:30,textAlign:"right"}}>{p.count}</span>
//         </div>
//       ))}
//     </div>
//   );
// }

// function StatsPage({stats, peopleFreq}){
//   if(!stats) return <Empty icon={BarChart3} msg="Loading stats…"/>;
//   const cards=[{l:"Photos",v:stats.total_images,c:"#818cf8"},{l:"Faces",v:stats.total_faces,c:"#a78bfa"},{l:"People",v:stats.total_people,c:"#f472b6"},{l:"Albums",v:stats.total_albums,c:"#fbbf24"},{l:"Favorites",v:stats.total_favorites,c:"#4ade80"},{l:"Indexed",v:stats.indexed_vectors,c:"#22d3ee"}];
//   return (
//     <div className="stats-wrap">
//       <PageHead title="Statistics"/>
//       <div className="stats-cards">{cards.map((c,i)=><motion.div key={i} className="stat-card" whileHover={{y:-3}} style={{"--sc":c.c}}><p className="stat-lbl">{c.l}</p><p className="stat-val" style={{color:c.c}}>{c.v??0}</p></motion.div>)}</div>

//       <div className="stats-sec">
//         <p className="stats-sec-lbl">EMOTION TIMELINE</p>
//         <EmotionTimelineChart/>
//       </div>

//       {peopleFreq?.length>0&&(
//         <div className="stats-sec">
//           <p className="stats-sec-lbl">MOST PHOTOGRAPHED PEOPLE</p>
//           <PeopleFreqChart people={peopleFreq}/>
//         </div>
//       )}

//       {stats.color_distribution?.length>0&&<div className="stats-sec"><p className="stats-sec-lbl">COLOR DISTRIBUTION</p><div className="clr-dist">{stats.color_distribution.map((d,i)=><div key={i} className="clr-dist-item"><div className="clr-dist-dot" style={{background:d.color==="gray"?"#888":d.color==="white"?"#ddd":d.color}}/><span>{d.color}</span><span className="clr-dist-n">×{d.count}</span></div>)}</div></div>}
//       {stats.top_user_tags?.length>0&&<div className="stats-sec"><p className="stats-sec-lbl">YOUR TAGS</p><div className="tag-cloud">{stats.top_user_tags.map((t,i)=><span key={i} className="tag-cloud-item" style={{borderColor:"#86efac44",color:"#86efac"}}>#{t.tag}<span className="tag-n" style={{color:"#86efac99"}}> ×{t.count}</span></span>)}</div></div>}
//       {stats.top_tags?.length>0&&<div className="stats-sec"><p className="stats-sec-lbl">AI DETECTED OBJECTS</p><div className="tag-cloud">{stats.top_tags.map((t,i)=><span key={i} className="tag-cloud-item">{t.tag}<span className="tag-n"> ×{t.count}</span></span>)}</div></div>}
//     </div>
//   );
// }

// function ToolsAccordion({reindexing, onEmotions, onColors, onNames, onCleanup}) {
//   const [open, setOpen] = React.useState(false);
//   return (
//     <div className="tools-accordion">
//       <button className="tools-accordion-btn" onClick={()=>setOpen(p=>!p)}>
//         <span style={{display:"flex",alignItems:"center",gap:6}}>
//           <SlidersHorizontal size={11}/> AI Tools
//         </span>
//         <motion.span animate={{rotate:open?180:0}} transition={{duration:.2}}>
//           <ChevronDown size={11}/>
//         </motion.span>
//       </button>
//       <AnimatePresence>
//         {open && (
//           <motion.div initial={{height:0,opacity:0}} animate={{height:"auto",opacity:1}} exit={{height:0,opacity:0}}
//             style={{overflow:"hidden"}}>
//             <div style={{display:"flex",flexDirection:"column",gap:4,paddingTop:4}}>
//               {[
//                 {label:"Fix Emotions",  color:"rgba(250,204,21,.8)",  bg:"rgba(250,204,21,.07)",  border:"rgba(250,204,21,.2)",  fn:onEmotions},
//                 {label:"Fix Colors",    color:"rgba(52,211,153,.85)", bg:"rgba(52,211,153,.07)",  border:"rgba(52,211,153,.2)",  fn:onColors},
//                 {label:"Auto-Name",     color:"rgba(96,165,250,.85)", bg:"rgba(96,165,250,.07)",  border:"rgba(96,165,250,.2)",  fn:onNames},
//                 {label:"Clean Albums",  color:"rgba(248,113,113,.8)", bg:"rgba(248,113,113,.07)", border:"rgba(248,113,113,.2)", fn:onCleanup},
//               ].map(({label,color,bg,border,fn})=>(
//                 <button key={label} disabled={reindexing}
//                   className="reindex-btn"
//                   style={{background:bg,borderColor:border,color,fontSize:10,padding:"7px"}}
//                   onClick={fn}>
//                   {label}
//                 </button>
//               ))}
//             </div>
//           </motion.div>
//         )}
//       </AnimatePresence>
//     </div>
//   );
// }

// function PageHead({title,count,unit="photos",extra}){ return <div className="page-head"><h1 className="page-title">{title}</h1>{count>0&&<span className="page-count">{count} {unit}</span>}{extra&&<div style={{marginLeft:"auto"}}>{extra}</div>}</div>; }
// function Empty({icon:Icon,msg,sub,onClick}){ return <div className={`empty ${onClick?"empty--click":""}`} onClick={onClick}><Icon size={44} strokeWidth={1} color="#333"/><p className="empty-msg">{msg}</p>{sub&&<p className="empty-sub">{sub}</p>}{onClick&&<p className="empty-cta">CLICK TO UPLOAD</p>}</div>; }
// function Fade({children}){ return <motion.div initial={{opacity:0,y:6}} animate={{opacity:1,y:0}} exit={{opacity:0}} transition={{duration:.18}}>{children}</motion.div>; }

// /* ══════════════════════════════════════════════════════════════════════════ */
// /* CSS                                                                         */
// /* ══════════════════════════════════════════════════════════════════════════ */
// function CSS() {
//   return <style>{`
// @import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300..900;1,14..32,300..900&family=JetBrains+Mono:wght@400;500&display=swap');

// *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

// :root{
//   /* Background layers */
//   --bg0:#000000;
//   --bg1:#0a0a0a;
//   --bg2:#111111;
//   --bg3:#161616;
//   --bg4:#1c1c1c;

//   /* Borders */
//   --b1:rgba(255,255,255,.06);
//   --b2:rgba(255,255,255,.1);
//   --b3:rgba(255,255,255,.15);

//   /* Accent */
//   --ac:#6366f1;
//   --ac-l:#818cf8;
//   --ac-d:#4f46e5;
//   --ac-glow:rgba(99,102,241,.25);
//   --ac-bg:rgba(99,102,241,.08);

//   /* Text */
//   --t1:#ffffff;
//   --t2:#a1a1aa;
//   --t3:#52525b;

//   /* Status */
//   --green:#22c55e;
//   --red:#ef4444;
//   --gold:#f59e0b;
//   --sky:#38bdf8;
//   --rose:#f43f5e;

//   --font:'Inter',system-ui,sans-serif;
//   --mono:'JetBrains Mono',monospace;
//   --r:8px;
//   --r2:12px;
//   --r3:16px;
// }

// html,body,#root{height:100%;background:var(--bg0);color:var(--t1);font-family:var(--font);overflow:hidden;-webkit-font-smoothing:antialiased}

// ::-webkit-scrollbar{width:3px;height:3px}
// ::-webkit-scrollbar-track{background:transparent}
// ::-webkit-scrollbar-thumb{background:var(--b2);border-radius:3px}

// /* ═══ LAYOUT ══════════════════════════════════════════════════════════════ */
// .shell{display:flex;height:100vh;overflow:hidden;background:var(--bg0)}

// /* ═══ SIDEBAR ═════════════════════════════════════════════════════════════ */
// .sidebar{
//   width:220px;flex-shrink:0;
//   background:var(--bg1);
//   border-right:1px solid var(--b1);
//   display:flex;flex-direction:column;
//   padding:0;overflow-y:auto;overflow-x:hidden;
// }

// /* Brand */
// .brand{
//   display:flex;align-items:center;gap:10px;
//   padding:20px 16px 16px;
//   border-bottom:1px solid var(--b1);
//   flex-shrink:0;
// }
// .brand-mark{
//   width:32px;height:32px;border-radius:8px;
//   background:var(--ac);
//   display:flex;align-items:center;justify-content:center;
//   flex-shrink:0;color:#fff;
//   box-shadow:0 0 0 1px rgba(99,102,241,.4),0 4px 12px rgba(99,102,241,.3);
// }
// .brand-name{font-size:14px;font-weight:700;letter-spacing:-.01em;color:var(--t1)}
// .brand-sub{font-family:var(--mono);font-size:9px;color:var(--t3);letter-spacing:.06em;margin-top:1px}

// /* Nav */
// .nav-list{display:flex;flex-direction:column;padding:8px;gap:1px;flex-shrink:0}
// .nav-btn{
//   display:flex;align-items:center;gap:9px;
//   width:100%;padding:8px 10px;
//   border-radius:var(--r);background:transparent;border:none;
//   font-family:var(--font);font-size:13px;font-weight:500;
//   color:var(--t3);cursor:pointer;text-align:left;
//   transition:color .12s,background .12s;position:relative;
// }
// .nav-btn:hover{background:var(--bg3);color:var(--t2)}
// .nav-btn--on{background:var(--bg3);color:var(--t1);font-weight:600}
// .nav-btn--on .nav-ic{color:var(--ac-l)}
// .nav-btn--trash:hover{color:var(--red);background:rgba(239,68,68,.06)}
// .nav-ic{display:flex;flex-shrink:0;color:inherit}
// .nav-pip{
//   position:absolute;right:8px;
//   width:4px;height:4px;border-radius:50%;background:var(--ac);
//   box-shadow:0 0 6px var(--ac);
// }
// .badge{
//   margin-left:auto;background:var(--red);color:#fff;
//   font-size:10px;font-family:var(--mono);
//   padding:1px 5px;border-radius:20px;line-height:1.5;
// }

// .sidebar-sep{height:1px;background:var(--b1);margin:6px 16px;flex-shrink:0}
// .sidebar-section-label{
//   display:flex;align-items:center;gap:5px;
//   font-size:10px;font-weight:600;color:var(--t3);
//   letter-spacing:.06em;text-transform:uppercase;
//   padding:8px 16px 4px;flex-shrink:0;
// }

// /* Tag nav */
// .tag-nav-list{display:flex;flex-direction:column;padding:0 8px;gap:1px}
// .tag-nav-btn{
//   display:flex;align-items:center;justify-content:space-between;
//   width:100%;padding:6px 10px;border-radius:var(--r);
//   background:transparent;border:none;
//   font-family:var(--mono);font-size:11px;color:var(--t3);
//   cursor:pointer;transition:all .12s;
// }
// .tag-nav-btn:hover{background:var(--bg3);color:var(--green)}
// .tag-nav-btn--on{background:rgba(34,197,94,.07);color:var(--green)}
// .tag-nav-count{font-size:9px;color:var(--t3)}

// /* Sidebar footer */
// .sidebar-foot{
//   display:flex;flex-direction:column;gap:4px;
//   margin-top:auto;padding:12px 8px;border-top:1px solid var(--b1);
//   flex-shrink:0;
// }
// .reindex-btn{
//   display:flex;align-items:center;justify-content:center;gap:6px;
//   background:transparent;border:1px solid var(--b1);
//   border-radius:var(--r);padding:8px;
//   font-family:var(--font);font-size:12px;font-weight:500;
//   color:var(--t3);cursor:pointer;transition:all .12s;
// }
// .reindex-btn:hover{border-color:var(--b2);color:var(--t2);background:var(--bg3)}
// .reindex-btn:disabled{opacity:.35;cursor:default}
// .reindex-btn--primary{
//   background:var(--ac-bg);border-color:rgba(99,102,241,.3);
//   color:var(--ac-l);font-weight:600;
// }
// .reindex-btn--primary:hover{background:rgba(99,102,241,.15);border-color:var(--ac)}

// .reindex-msg{font-family:var(--mono);font-size:9px;color:var(--t3);text-align:center;padding:2px 4px;line-height:1.5}

// .offline-pill{
//   display:flex;align-items:center;gap:6px;
//   background:rgba(34,197,94,.05);border:1px solid rgba(34,197,94,.12);
//   border-radius:var(--r);padding:7px 10px;
//   font-size:11px;font-weight:500;color:rgba(34,197,94,.7);
// }
// .dot-green{width:6px;height:6px;border-radius:50%;background:var(--green);box-shadow:0 0 6px var(--green);flex-shrink:0}

// /* Tools accordion */
// .tools-accordion{border:1px solid var(--b1);border-radius:var(--r);overflow:hidden;background:var(--bg2)}
// .tools-accordion-btn{
//   display:flex;align-items:center;justify-content:space-between;
//   width:100%;padding:8px 10px;background:transparent;border:none;
//   font-family:var(--font);font-size:11px;font-weight:500;color:var(--t3);cursor:pointer;
//   transition:color .12s;
// }
// .tools-accordion-btn:hover{color:var(--t2)}

// /* ═══ MAIN ════════════════════════════════════════════════════════════════ */
// .main{flex:1;display:flex;flex-direction:column;min-width:0;overflow:hidden}

// /* ═══ TOPBAR ══════════════════════════════════════════════════════════════ */
// .topbar{
//   background:var(--bg1);
//   border-bottom:1px solid var(--b1);
//   padding:10px 20px;
//   display:flex;flex-direction:column;gap:8px;flex-shrink:0;
// }

// /* Mode pills */
// .mode-strip{display:flex;gap:2px;align-items:center;flex-wrap:wrap}
// .mode-btn{
//   display:flex;align-items:center;gap:5px;
//   padding:5px 10px;border-radius:6px;border:none;cursor:pointer;
//   font-family:var(--font);font-size:11px;font-weight:500;
//   background:transparent;color:var(--t3);transition:all .12s;
// }
// .mode-btn:hover{background:var(--bg3);color:var(--t2)}
// .mode-btn--on{background:var(--ac);color:#fff;font-weight:600}
// .voice-chip{
//   margin-left:6px;display:flex;align-items:center;gap:4px;
//   font-family:var(--mono);font-size:9px;letter-spacing:.05em;
//   padding:4px 10px;border-radius:6px;
// }
// .voice-chip--rec{background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.2);color:var(--red)}
// .voice-chip--processing{background:var(--ac-bg);border:1px solid rgba(99,102,241,.2);color:var(--ac-l)}
// .voice-chip--error{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.15);color:#f87171}

// /* Search bar */
// .search-bar{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
// .search-wrap{
//   flex:1;min-width:200px;
//   display:flex;align-items:center;gap:8px;
//   background:var(--bg2);border:1px solid var(--b1);
//   border-radius:var(--r2);padding:0 12px;
//   transition:border-color .15s;
// }
// .search-wrap:focus-within{border-color:rgba(99,102,241,.4)}
// .search-ic{color:var(--t3);flex-shrink:0}
// .search-in{
//   flex:1;background:transparent;border:none;
//   padding:9px 0;font-family:var(--font);font-size:13px;
//   color:var(--t1);outline:none;
// }
// .search-in::placeholder{color:var(--t3)}
// .mic-btn{background:transparent;border:none;cursor:pointer;color:var(--t3);padding:3px;display:flex;border-radius:5px;transition:color .12s}
// .mic-btn:hover{color:var(--t2)}
// .mic-btn--rec{color:var(--red)}
// .mic-btn--proc{color:var(--ac-l)}

// .img-search-label{
//   display:flex;align-items:center;gap:7px;cursor:pointer;
//   background:var(--bg2);border:1px solid var(--b1);
//   border-radius:var(--r2);padding:8px 12px;
//   font-size:12px;color:var(--t3);flex-shrink:0;transition:all .12s;
// }
// .img-search-label:hover{border-color:var(--b2);color:var(--t2)}
// .img-search-thumb{width:20px;height:20px;border-radius:4px;object-fit:cover}

// .color-strip{display:flex;gap:5px;align-items:center;flex-wrap:wrap}
// .clr-dot{
//   width:22px;height:22px;border-radius:50%;border:none;cursor:pointer;
//   outline:2px solid transparent;outline-offset:2px;transition:all .12s;flex-shrink:0;
// }
// .clr-dot:hover{transform:scale(1.15)}
// .clr-dot--on{outline-color:rgba(255,255,255,.5);transform:scale(1.2)}

// .btn-go{
//   background:var(--ac);color:#fff;border:none;border-radius:var(--r2);
//   padding:8px 16px;font-family:var(--font);font-size:12px;font-weight:600;
//   cursor:pointer;transition:background .12s;flex-shrink:0;
// }
// .btn-go:hover{background:var(--ac-d)}

// .btn-batch{
//   display:flex;align-items:center;gap:5px;
//   background:var(--bg2);border:1px solid var(--b1);
//   color:var(--t3);border-radius:var(--r2);
//   padding:8px 12px;font-family:var(--font);font-size:12px;font-weight:500;
//   cursor:pointer;transition:all .12s;flex-shrink:0;
// }
// .btn-batch:hover{border-color:var(--b2);color:var(--t2)}
// .btn-batch--on{background:var(--ac-bg);border-color:rgba(99,102,241,.3);color:var(--ac-l)}

// .btn-upload{
//   display:flex;align-items:center;gap:5px;
//   background:var(--bg2);border:1px solid var(--b2);
//   color:var(--t2);border-radius:var(--r2);
//   padding:8px 14px;font-family:var(--font);font-size:12px;font-weight:600;
//   cursor:pointer;transition:all .12s;flex-shrink:0;
// }
// .btn-upload:hover{border-color:var(--b3);color:var(--t1);background:var(--bg3)}

// /* Emotion filter bar */
// .emo-filter-bar{display:flex;align-items:center;gap:5px;flex-wrap:wrap}
// .emo-filter-label{font-family:var(--mono);font-size:9px;color:var(--t3);letter-spacing:.06em;flex-shrink:0}
// .emo-filter-btn{
//   display:flex;align-items:center;gap:4px;
//   padding:4px 10px;border-radius:20px;
//   border:1px solid var(--b1);background:transparent;
//   color:var(--t3);font-family:var(--font);font-size:11px;font-weight:500;
//   cursor:pointer;transition:all .12s;flex-shrink:0;
// }
// .emo-filter-btn:hover{border-color:var(--b2);color:var(--t2)}
// .emo-filter-count{font-family:var(--mono);font-size:8px;opacity:.5;margin-left:2px}
// .emo-filter-clear{
//   display:flex;align-items:center;gap:4px;padding:4px 9px;
//   border-radius:20px;border:1px solid rgba(239,68,68,.2);
//   background:rgba(239,68,68,.06);color:var(--red);
//   font-family:var(--font);font-size:11px;font-weight:500;cursor:pointer;transition:all .12s;
// }

// /* Emoji bar */
// .emoji-quick-bar{display:flex;align-items:center;gap:3px;flex-wrap:wrap}
// .emoji-quick-label{font-family:var(--mono);font-size:9px;color:var(--t3);letter-spacing:.06em;flex-shrink:0;margin-right:2px}
// .emoji-quick-btn{
//   background:var(--bg3);border:1px solid var(--b1);
//   border-radius:6px;padding:3px 6px;font-size:14px;
//   cursor:pointer;transition:all .12s;line-height:1;
// }
// .emoji-quick-btn:hover{background:var(--ac-bg);border-color:rgba(99,102,241,.3);transform:scale(1.12)}

// /* ═══ BODY ════════════════════════════════════════════════════════════════ */
// .body{flex:1;overflow-y:auto;padding:20px 20px 80px}
// .page-head{display:flex;align-items:baseline;gap:10px;margin-bottom:18px;flex-wrap:wrap}
// .page-title{font-size:18px;font-weight:700;letter-spacing:-.025em;color:var(--t1)}
// .page-count{font-family:var(--mono);font-size:9px;color:var(--t3);letter-spacing:.05em}
// .sel-count{font-family:var(--mono);font-size:9px;color:var(--ac-l);background:var(--ac-bg);padding:2px 8px;border-radius:20px;border:1px solid rgba(99,102,241,.2)}

// /* On This Day */
// .otd-banner{
//   background:rgba(245,158,11,.04);
//   border:1px solid rgba(245,158,11,.12);
//   border-radius:var(--r3);margin-bottom:18px;overflow:hidden;
// }
// .otd-header{display:flex;align-items:center;justify-content:space-between;padding:12px 16px;cursor:pointer}
// .otd-left{display:flex;align-items:center;gap:10px}
// .otd-title{font-size:13px;font-weight:600;display:block}
// .otd-sub{font-family:var(--mono);font-size:9px;color:rgba(245,158,11,.5);display:block;margin-top:1px}
// .otd-toggle{background:none;border:none;cursor:pointer;color:var(--t3);display:flex;padding:3px;border-radius:5px}
// .otd-year{padding:0 16px 14px}
// .otd-year-label{font-family:var(--mono);font-size:9px;color:rgba(245,158,11,.5);letter-spacing:.06em;margin-bottom:8px;display:block}
// .otd-strip{display:flex;gap:8px;overflow-x:auto;padding-bottom:4px}
// .otd-thumb{flex-shrink:0;width:96px;cursor:pointer;border-radius:var(--r);overflow:hidden;border:1px solid var(--b1);transition:border-color .15s}
// .otd-thumb:hover{border-color:rgba(245,158,11,.3)}
// .otd-thumb img{width:100%;height:64px;object-fit:cover;display:block}
// .otd-thumb-caption{font-size:9px;color:var(--t3);padding:4px 6px;line-height:1.4;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}

// /* ═══ IMAGE GRID ══════════════════════════════════════════════════════════ */
// .img-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(176px,1fr));gap:8px}
// .img-grid--sm{grid-template-columns:repeat(auto-fill,minmax(124px,1fr));gap:6px}

// /* Photo card */
// .photo-card{
//   position:relative;aspect-ratio:1;
//   border-radius:var(--r2);overflow:hidden;cursor:pointer;
//   background:var(--bg2);border:1px solid var(--b1);
//   transition:border-color .15s;
// }
// .photo-card:hover{border-color:var(--b2)}
// .photo-card--sel{border-color:var(--ac)!important;box-shadow:0 0 0 2px rgba(99,102,241,.3)!important}
// .photo-img{width:100%;height:100%;object-fit:cover;display:block;transition:transform .25s}
// .photo-card:hover .photo-img{transform:scale(1.04)}
// .chip{position:absolute;font-family:var(--mono);font-size:8px;border-radius:4px;border:1px solid;padding:1px 5px;pointer-events:none;backdrop-filter:blur(4px)}
// .chip--q{top:7px;left:7px}
// .chip--emo{top:7px;right:7px;font-size:13px;border:none;background:none;padding:0}
// .fav-dot{position:absolute;bottom:7px;right:7px;color:var(--rose);font-size:13px;pointer-events:none}
// .batch-check{position:absolute;top:7px;left:7px}
// .photo-hover{
//   position:absolute;inset:0;opacity:0;transition:opacity .18s;
//   background:linear-gradient(to top,rgba(0,0,0,.9) 0%,transparent 55%);
//   display:flex;flex-direction:column;justify-content:flex-end;
// }
// .photo-card:hover .photo-hover{opacity:1}
// .photo-actions{position:absolute;top:7px;right:7px;display:flex;gap:4px}
// .ph-btn{
//   width:28px;height:28px;border-radius:6px;border:none;cursor:pointer;
//   display:flex;align-items:center;justify-content:center;transition:all .12s;
//   backdrop-filter:blur(6px);
// }
// .ph-btn--fav{background:rgba(244,63,94,.7);color:#fff}
// .ph-btn--del{background:rgba(0,0,0,.6);border:1px solid rgba(255,255,255,.1);color:#888}
// .ph-btn--del:hover{background:rgba(239,68,68,.8);color:#fff;border-color:transparent}
// .photo-info{padding:8px 8px 6px}
// .photo-caption{font-size:10px;line-height:1.45;color:rgba(255,255,255,.9);margin-bottom:4px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
// .photo-meta-row{display:flex;flex-wrap:wrap;gap:3px}
// .meta-chip{font-family:var(--mono);font-size:7.5px;color:rgba(255,255,255,.45);background:rgba(0,0,0,.4);padding:1px 5px;border-radius:3px;text-transform:capitalize}

// /* ═══ BATCH BAR ═══════════════════════════════════════════════════════════ */
// .batch-bar{
//   position:absolute;bottom:0;left:0;right:0;
//   background:rgba(10,10,10,.97);backdrop-filter:blur(12px);
//   border-top:1px solid var(--b1);
//   padding:10px 20px;
//   display:flex;align-items:center;gap:10px;flex-wrap:wrap;z-index:50;
// }
// .batch-bar-left{display:flex;align-items:center;gap:7px}
// .batch-count{display:flex;align-items:center;gap:5px;font-size:12px;font-weight:600;color:var(--ac-l)}
// .batch-sm{background:var(--bg3);border:1px solid var(--b1);color:var(--t3);border-radius:6px;padding:4px 10px;font-family:var(--font);font-size:11px;font-weight:500;cursor:pointer;transition:all .12s}
// .batch-sm:hover{border-color:var(--b2);color:var(--t2)}
// .batch-bar-actions{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-left:auto}
// .batch-action-btn{
//   display:flex;align-items:center;gap:5px;padding:6px 12px;
//   border-radius:var(--r);border:1px solid var(--b1);
//   background:var(--bg2);color:var(--t2);
//   font-family:var(--font);font-size:11px;font-weight:500;cursor:pointer;transition:all .12s;
// }
// .batch-action-btn:hover{border-color:var(--b2);color:var(--t1);background:var(--bg3)}
// .batch-action-btn--fav:hover{border-color:rgba(244,63,94,.3);color:var(--rose)}
// .batch-action-btn--del:hover{border-color:rgba(239,68,68,.3);color:var(--red)}
// .batch-action-btn--album{border-color:rgba(56,189,248,.2);color:var(--sky)}
// .batch-action-btn--album:hover{background:rgba(56,189,248,.08);border-color:rgba(56,189,248,.3)}
// .batch-input-group{display:flex;gap:3px;align-items:center}
// .batch-input{background:var(--bg2);border:1px solid var(--b1);border-radius:var(--r);padding:6px 10px;font-family:var(--font);font-size:11px;color:var(--t1);outline:none;width:108px;transition:border-color .12s}
// .batch-input:focus{border-color:rgba(99,102,241,.4)}
// .batch-select{background:var(--bg2);border:1px solid var(--b1);border-radius:var(--r);padding:6px 10px;font-family:var(--font);font-size:11px;color:var(--t2);outline:none;cursor:pointer}
// .batch-select:focus{border-color:rgba(99,102,241,.4)}

// /* ═══ LIGHTBOX ════════════════════════════════════════════════════════════ */
// .lb-bg{
//   position:fixed;inset:0;z-index:300;
//   background:rgba(0,0,0,.9);backdrop-filter:blur(16px);
//   display:flex;align-items:center;justify-content:center;padding:14px;
// }
// .lb-box{
//   width:100%;max-width:1100px;max-height:92vh;
//   background:var(--bg1);border:1px solid var(--b1);
//   border-radius:16px;display:grid;grid-template-columns:1fr 300px;
//   overflow:hidden;box-shadow:0 24px 80px rgba(0,0,0,.8);
// }
// .lb-left{position:relative;background:var(--bg0);display:flex;align-items:center;justify-content:center;min-height:380px;overflow:hidden}
// .lb-img{max-width:100%;max-height:90vh;object-fit:contain;display:block}
// .lb-nav{
//   position:absolute;top:50%;transform:translateY(-50%);
//   width:38px;height:38px;border-radius:50%;border:1px solid var(--b2);
//   cursor:pointer;background:rgba(0,0,0,.5);backdrop-filter:blur(6px);
//   color:#fff;display:flex;align-items:center;justify-content:center;
//   transition:all .15s;z-index:2;
// }
// .lb-nav:hover{background:var(--ac);border-color:transparent}
// .lb-nav--l{left:12px}
// .lb-nav--r{right:12px}
// .lb-counter{position:absolute;bottom:12px;left:50%;transform:translateX(-50%);font-family:var(--mono);font-size:9px;color:rgba(255,255,255,.35);background:rgba(0,0,0,.5);padding:2px 10px;border-radius:20px}
// .lb-hint{position:absolute;bottom:12px;right:14px;font-family:var(--mono);font-size:7.5px;color:rgba(255,255,255,.15);letter-spacing:.05em}
// .lb-right{border-left:1px solid var(--b1);background:var(--bg1);display:flex;flex-direction:column;min-height:0}
// .lb-right-head{display:flex;align-items:center;justify-content:space-between;padding:14px 16px;border-bottom:1px solid var(--b1);flex-shrink:0}
// .lb-right-title{font-size:13px;font-weight:600;letter-spacing:-.01em}
// .lb-head-btns{display:flex;align-items:center;gap:3px}
// .lb-hbtn{background:none;border:none;cursor:pointer;color:var(--t3);padding:6px;border-radius:6px;display:flex;align-items:center;transition:all .12s}
// .lb-hbtn:hover{background:var(--bg3);color:var(--t2)}
// .lb-close:hover{color:var(--red)}
// .lb-scroll{flex:1;overflow-y:auto;padding:12px 14px;display:flex;flex-direction:column;gap:7px}

// /* Detail blocks */
// .db{background:var(--bg2);border:1px solid var(--b1);border-radius:var(--r2);overflow:hidden;border-left:2px solid var(--dc,var(--ac))}
// .db-head{display:flex;align-items:center;gap:6px;padding:7px 11px;border-bottom:1px solid var(--b1)}
// .db-label{font-family:var(--mono);font-size:8px;color:var(--dc,var(--ac-l));letter-spacing:.08em;font-weight:500;text-transform:uppercase}
// .db-body{padding:9px 11px}
// .db-caption{font-size:12px;line-height:1.55;color:var(--t1)}
// .db-caption-sub{font-size:11px;line-height:1.55;color:var(--t2);margin-top:4px}
// .q-wrap{display:flex;align-items:flex-start;gap:10px}
// .q-circle{width:42px;height:42px;border-radius:50%;border:2px solid;display:flex;align-items:center;justify-content:center;flex-shrink:0}
// .q-right{flex:1}
// .q-level{font-size:12px;font-weight:600;margin-bottom:6px}
// .qbar-row{display:flex;align-items:center;gap:7px;margin-bottom:4px}
// .qbar-label{font-family:var(--mono);font-size:7.5px;color:var(--t3);width:65px;flex-shrink:0}
// .qbar-track{flex:1;height:2px;background:rgba(255,255,255,.07);border-radius:2px;overflow:hidden}
// .qbar-fill{height:100%;border-radius:2px}
// .qbar-num{font-family:var(--mono);font-size:7.5px;color:var(--t3);width:22px;text-align:right}
// .aes-row{display:flex;align-items:center;gap:9px}
// .aes-score{font-size:24px;font-weight:800;color:var(--gold)}
// .aes-rating{font-size:11px;color:var(--t2)}
// .emo-row{display:flex;align-items:center;gap:10px}
// .emo-name{font-size:12px;font-weight:600;text-transform:capitalize}
// .emo-sub{font-family:var(--mono);font-size:8px;color:var(--t3);margin-top:2px}
// .ocr-text{font-family:var(--mono);font-size:9px;color:var(--t2);line-height:1.6;white-space:pre-wrap;max-height:84px;overflow-y:auto}
// .tag-wrap{display:flex;flex-wrap:wrap;gap:4px}
// .obj-tag{font-family:var(--mono);font-size:8.5px;background:var(--bg3);color:var(--t3);padding:2px 7px;border-radius:4px;border:1px solid var(--b1)}
// .meta-stack{display:flex;flex-direction:column;gap:5px}
// .mrow{display:flex;justify-content:space-between;align-items:baseline;gap:12px}
// .mrow-k{font-family:var(--mono);font-size:8px;color:var(--t3);flex-shrink:0}
// .mrow-v{font-size:11px;color:var(--t1);text-align:right}

// /* Tags */
// .tag-chips{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px}
// .tag-chip-item{display:flex;align-items:center;gap:3px;background:rgba(34,197,94,.07);border:1px solid rgba(34,197,94,.15);color:var(--green);font-family:var(--mono);font-size:8.5px;padding:2px 7px;border-radius:4px}
// .tag-chip-x{background:none;border:none;cursor:pointer;color:rgba(34,197,94,.4);display:flex;padding:0;margin-left:2px;transition:color .12s}
// .tag-chip-x:hover{color:var(--green)}
// .tag-add-row{display:flex;gap:4px;align-items:center}
// .tag-add-input{flex:1;background:var(--bg3);border:1px solid var(--b1);border-radius:6px;padding:5px 9px;font-family:var(--mono);font-size:10px;color:var(--t1);outline:none;transition:border-color .12s}
// .tag-add-input:focus{border-color:rgba(99,102,241,.4)}
// .tag-add-btn{background:var(--bg3);border:1px solid var(--b1);border-radius:6px;padding:5px 8px;cursor:pointer;color:var(--t3);display:flex;transition:all .12s}
// .tag-add-btn:hover{border-color:rgba(99,102,241,.3);color:var(--ac-l)}

// /* ═══ TRASH ═══════════════════════════════════════════════════════════════ */
// .trash-card{position:relative;aspect-ratio:1;border-radius:var(--r2);overflow:hidden;background:var(--bg2);border:1px solid var(--b1)}
// .trash-img{width:100%;height:100%;object-fit:cover;filter:grayscale(.5) brightness(.4)}
// .trash-overlay{position:absolute;inset:0;opacity:0;transition:opacity .18s;background:rgba(0,0,0,.7);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:6px;padding:10px}
// .trash-card:hover .trash-overlay{opacity:1}
// .trash-btn{width:100%;display:flex;align-items:center;justify-content:center;gap:5px;border:none;border-radius:var(--r);padding:7px;font-family:var(--font);font-size:11px;font-weight:600;cursor:pointer;transition:all .12s}
// .trash-btn--restore{background:var(--ac);color:#fff}
// .trash-btn--del{background:rgba(220,50,50,.85);color:#fff}
// .trash-date{position:absolute;top:7px;left:7px;font-family:var(--mono);font-size:7.5px;background:rgba(0,0,0,.7);color:var(--t3);padding:2px 6px;border-radius:4px}

// /* ═══ PEOPLE ══════════════════════════════════════════════════════════════ */
// .people-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(108px,1fr));gap:18px}
// .person-card{display:flex;flex-direction:column;align-items:center;gap:8px;cursor:pointer}
// .person-avatar{width:68px;height:68px;border-radius:50%;overflow:hidden;border:1px solid var(--b1);background:var(--bg2);display:flex;align-items:center;justify-content:center;transition:border-color .15s}
// .person-avatar img{width:100%;height:100%;object-fit:cover}
// .person-card:hover .person-avatar{border-color:var(--ac)}
// .person-name{font-size:12px;font-weight:600;text-align:center}
// .person-count{font-family:var(--mono);font-size:8px;color:var(--t3)}

// /* ═══ ALBUMS ══════════════════════════════════════════════════════════════ */
// .albums-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px}
// .album-card{
//   position:relative;aspect-ratio:16/10;
//   border-radius:var(--r3);overflow:hidden;cursor:pointer;
//   background:var(--bg2);border:1px solid var(--b1);
//   transition:border-color .15s;
// }
// .album-card:hover{border-color:var(--b2)}
// .album-cover{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;transition:transform .3s}
// .album-card:hover .album-cover{transform:scale(1.04)}
// .album-cover-ph{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;color:var(--t3)}
// .album-strip{position:absolute;bottom:0;left:0;right:0;height:28px;display:flex;opacity:0;transition:opacity .18s;z-index:2}
// .album-card:hover .album-strip{opacity:1}
// .album-strip-img{flex:1;height:100%;object-fit:cover;border-right:1px solid rgba(0,0,0,.3)}
// .album-info{position:absolute;inset:0;z-index:1;background:linear-gradient(to top,rgba(0,0,0,.9) 0%,transparent 55%);padding:12px;display:flex;flex-direction:column;justify-content:flex-end}
// .album-title{font-weight:700;font-size:13px;letter-spacing:-.01em;margin-bottom:5px;line-height:1.3;color:#fff}
// .album-meta{display:flex;align-items:center;gap:5px;flex-wrap:wrap}
// .album-del-btn{position:absolute;top:8px;right:8px;width:24px;height:24px;border-radius:50%;background:rgba(0,0,0,.65);border:1px solid var(--b2);color:var(--t3);display:flex;align-items:center;justify-content:center;cursor:pointer;z-index:3;opacity:0;transition:all .15s}
// .album-card:hover .album-del-btn{opacity:1}
// .album-del-btn:hover{background:rgba(220,50,50,.85);color:#fff;border-color:transparent}
// .tag-sm{font-family:var(--mono);font-size:8px;background:rgba(255,255,255,.08);color:var(--t3);padding:2px 6px;border-radius:20px}
// .tag-sm--accent{background:rgba(99,102,241,.15);color:var(--ac-l)}
// .tag-sm--manual{background:rgba(245,158,11,.1);color:rgba(245,158,11,.8)}

// /* ═══ DUPLICATES ══════════════════════════════════════════════════════════ */
// .dupe-list{display:flex;flex-direction:column;gap:14px}
// .dupe-group{background:var(--bg2);border:1px solid var(--b1);border-radius:var(--r3);padding:16px}
// .dupe-head{display:flex;align-items:center;gap:7px;margin-bottom:14px;flex-wrap:wrap}

// /* ═══ MODAL ═══════════════════════════════════════════════════════════════ */
// .modal-bg{position:fixed;inset:0;z-index:200;background:rgba(0,0,0,.8);backdrop-filter:blur(10px);display:flex;align-items:center;justify-content:center;padding:18px}
// .modal-box{width:100%;max-height:90vh;overflow-y:auto;background:var(--bg1);border:1px solid var(--b1);border-radius:var(--r3);padding:20px;box-shadow:0 16px 50px rgba(0,0,0,.7)}
// .modal-head{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:14px;gap:10px}
// .modal-title{font-size:18px;font-weight:700;letter-spacing:-.025em}
// .modal-sub{font-family:var(--mono);font-size:8px;color:var(--t3);letter-spacing:.06em;margin-top:3px}
// .rename-row{display:flex;gap:7px;margin-bottom:14px;align-items:center}
// .rename-in{flex:1;background:var(--bg2);border:1px solid var(--b1);border-radius:var(--r);padding:8px 11px;font-family:var(--font);font-size:13px;color:var(--t1);outline:none;transition:border-color .12s}
// .rename-in:focus{border-color:rgba(99,102,241,.4)}
// .album-desc{background:var(--ac-bg);border:1px solid rgba(99,102,241,.15);border-radius:var(--r);padding:10px 12px;margin-bottom:14px}
// .album-desc-lbl{font-family:var(--mono);font-size:8px;color:var(--ac-l);letter-spacing:.08em;display:block;margin-bottom:4px}
// .album-desc p{font-size:12px;line-height:1.6}
// .create-album-form{display:flex;flex-direction:column;gap:9px}
// .form-input{background:var(--bg2);border:1px solid var(--b1);border-radius:var(--r);padding:10px 12px;font-family:var(--font);font-size:13px;color:var(--t1);outline:none;transition:border-color .12s}
// .form-input:focus{border-color:rgba(99,102,241,.4)}
// .form-textarea{background:var(--bg2);border:1px solid var(--b1);border-radius:var(--r);padding:10px 12px;font-family:var(--font);font-size:13px;color:var(--t1);outline:none;resize:vertical;transition:border-color .12s}
// .form-textarea:focus{border-color:rgba(99,102,241,.4)}

// .btn-sm{display:inline-flex;align-items:center;gap:5px;background:var(--bg2);border:1px solid var(--b1);color:var(--t2);border-radius:var(--r);padding:6px 12px;font-family:var(--font);font-size:11px;font-weight:500;cursor:pointer;transition:all .12s}
// .btn-sm:hover{border-color:var(--b2);color:var(--t1);background:var(--bg3)}
// .btn-sm--primary{background:var(--ac);border-color:transparent;color:#fff}
// .btn-sm--primary:hover{background:var(--ac-d)}
// .btn-sm--accent{background:var(--ac-bg);border-color:rgba(99,102,241,.25);color:var(--ac-l)}
// .btn-sm--accent:hover{background:var(--ac);color:#fff;border-color:transparent}
// .btn-sm:disabled{opacity:.35;cursor:default}

// /* ═══ STATS ═══════════════════════════════════════════════════════════════ */
// .stats-wrap{max-width:820px}
// .stats-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:14px}
// .stat-card{
//   background:var(--bg2);border:1px solid var(--b1);
//   border-radius:var(--r2);padding:16px 18px;
//   border-top:2px solid var(--sc);
//   transition:border-color .15s;
// }
// .stat-card:hover{border-color:var(--b2)}
// .stat-lbl{font-family:var(--mono);font-size:8px;color:var(--t3);letter-spacing:.09em;margin-bottom:8px;text-transform:uppercase}
// .stat-val{font-size:36px;font-weight:800;letter-spacing:-.04em;line-height:1;color:var(--sc)}
// .stats-sec{background:var(--bg2);border:1px solid var(--b1);border-radius:var(--r2);padding:14px;margin-bottom:8px}
// .stats-sec-lbl{font-family:var(--mono);font-size:8px;color:var(--t3);letter-spacing:.09em;margin-bottom:12px;text-transform:uppercase}
// .clr-dist{display:flex;flex-wrap:wrap;gap:6px}
// .clr-dist-item{display:flex;align-items:center;gap:6px;background:var(--bg3);border:1px solid var(--b1);border-radius:6px;padding:5px 10px;font-size:11px}
// .clr-dist-dot{width:10px;height:10px;border-radius:50%;border:1px solid rgba(255,255,255,.1);flex-shrink:0}
// .clr-dist-n{font-family:var(--mono);font-size:9px;color:var(--t3)}
// .tag-cloud{display:flex;flex-wrap:wrap;gap:5px}
// .tag-cloud-item{font-family:var(--mono);font-size:9px;background:var(--ac-bg);color:var(--ac-l);border:1px solid rgba(99,102,241,.15);padding:3px 9px;border-radius:20px}
// .tag-n{opacity:.5}

// /* ═══ EMPTY / LOADER ══════════════════════════════════════════════════════ */
// .empty{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:240px;background:var(--bg2);border:1px dashed var(--b1);border-radius:var(--r3);padding:40px;gap:10px;text-align:center}
// .empty--click{cursor:pointer;transition:background .12s}
// .empty--click:hover{background:var(--bg3)}
// .empty-msg{font-size:16px;font-weight:700;letter-spacing:-.02em}
// .empty-sub{font-size:12px;color:var(--t2);max-width:240px;line-height:1.6}
// .empty-cta{font-family:var(--mono);font-size:9px;color:var(--ac-l);letter-spacing:.09em;margin-top:4px;text-transform:uppercase}
// .loader-wrap{display:flex;flex-direction:column;align-items:center;justify-content:center;height:55vh;gap:12px}
// .loader-ring{width:34px;height:34px;border-radius:50%;border:2px solid rgba(99,102,241,.12);border-top-color:var(--ac);animation:spin .7s linear infinite}
// .loader-text{font-family:var(--mono);font-size:10px;color:var(--t3);letter-spacing:.1em}
// @keyframes spin{to{transform:rotate(360deg)}}
// .spin{animation:spin .7s linear infinite}

// /* ═══ MISC ════════════════════════════════════════════════════════════════ */
// .person-chip{display:flex;align-items:center;padding:5px 11px;background:var(--bg2);border:1px solid var(--b1);border-radius:20px;font-size:12px;color:var(--t3);cursor:pointer;transition:all .12s}
// .person-chip:hover{border-color:rgba(99,102,241,.3);color:var(--t2)}
// .person-chip--on{background:rgba(49,46,129,.4);border-color:var(--ac);color:var(--ac-l)}

// @media(max-width:720px){
//   .lb-box{grid-template-columns:1fr}
//   .lb-right{border-left:none;border-top:1px solid var(--b1);max-height:40vh}
//   .stats-cards{grid-template-columns:repeat(2,1fr)}
//   .sidebar{width:180px}
// }
// `}</style>;
// }

import React, { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Search, Upload, Image as ImgIcon, Clock, BookOpen, Users, Heart,
  Copy, Trash2, X, ChevronLeft, ChevronRight, Sparkles, BarChart3,
  GitMerge, Palette, Shuffle, RotateCcw, AlertTriangle, FileText,
  Mic, MicOff, Zap, Star, Eye, Camera, Info, Tag, Type, Smile,
  Loader2, Trash, BookImage, ImageOff, TrendingUp, Aperture,
  CheckSquare, Square, Plus, Calendar, Hash, FolderPlus,
  CheckCheck, Layers, ChevronDown, ChevronUp,
  SlidersHorizontal, UserCheck
} from "lucide-react";
import axios from "axios";

const API = "http://localhost:8000";
const imgUrl = (f) => {
  if (!f) return null;
  const b = String(f).split("/").filter(Boolean).pop();
  return b ? `${API}/image/${b}` : null;
};
const BLANK = `data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='1' height='1'%3E%3Crect fill='%23111118'/%3E%3C/svg%3E`;

const QL = {
  Excellent: { c:"#4ade80", bg:"rgba(74,222,128,.12)" },
  Good:      { c:"#818cf8", bg:"rgba(129,140,248,.12)" },
  Fair:      { c:"#fbbf24", bg:"rgba(251,191,36,.12)"  },
  Poor:      { c:"#f87171", bg:"rgba(248,113,113,.12)" },
};
const EMO = { happy:"😊",sad:"😢",angry:"😠",surprised:"😲",disgusted:"🤢",fearful:"😨",neutral:"😐" };
const EMO_COLORS = { happy:"#fbbf24",sad:"#60a5fa",angry:"#f87171",surprised:"#a78bfa",disgusted:"#4ade80",fearful:"#fb923c",neutral:"#6b7280" };
const qInfo = (l) => QL[l] || { c:"#555", bg:"rgba(255,255,255,.05)" };

/* ─── Voice hook ─────────────────────────────────────────────────────────── */
function useVoice(onResult, onError) {
  const [state, setState] = useState("idle");
  const streamRef = useRef(null);
  const processorRef = useRef(null);
  const ctxRef = useRef(null);
  const samplesRef = useRef([]);
  const ok = typeof navigator !== "undefined" && !!navigator.mediaDevices?.getUserMedia;

  // Convert Float32 PCM samples → 16-bit WAV blob (no ffmpeg needed)
  const buildWav = (samples, sampleRate) => {
    const len = samples.length;
    const buf = new ArrayBuffer(44 + len * 2);
    const view = new DataView(buf);
    const write = (off, str) => { for(let i=0;i<str.length;i++) view.setUint8(off+i, str.charCodeAt(i)); };
    write(0,"RIFF"); view.setUint32(4, 36+len*2, true);
    write(8,"WAVE"); write(12,"fmt ");
    view.setUint32(16,16,true); view.setUint16(20,1,true); view.setUint16(22,1,true);
    view.setUint32(24,sampleRate,true); view.setUint32(28,sampleRate*2,true);
    view.setUint16(32,2,true); view.setUint16(34,16,true);
    write(36,"data"); view.setUint32(40,len*2,true);
    for(let i=0;i<len;i++){
      const s = Math.max(-1,Math.min(1,samples[i]));
      view.setInt16(44+i*2, s<0?s*0x8000:s*0x7FFF, true);
    }
    return new Blob([buf], {type:"audio/wav"});
  };

  const start = useCallback(async () => {
    if (!ok) return onError?.("Mic unavailable");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true }
      });
      streamRef.current = stream;
      samplesRef.current = [];

      // Use Web Audio API to capture raw PCM at 16kHz — bypasses ffmpeg completely
      const ctx = new AudioContext({ sampleRate: 16000 });
      ctxRef.current = ctx;
      const source = ctx.createMediaStreamSource(stream);
      const processor = ctx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = e => {
        const data = e.inputBuffer.getChannelData(0);
        samplesRef.current.push(new Float32Array(data));
      };
      source.connect(processor);
      processor.connect(ctx.destination);
      setState("rec");
    } catch(e) {
      setState("error");
      onError?.(e.name === "NotAllowedError" ? "Mic permission denied" : "Mic unavailable");
      setTimeout(()=>setState("idle"),3000);
    }
  }, [ok]);

  const stop = useCallback(async () => {
    if (state !== "rec") return;
    setState("processing");
    try {
      // Stop all tracks
      streamRef.current?.getTracks().forEach(t=>t.stop());
      processorRef.current?.disconnect();
      await ctxRef.current?.close();

      // Flatten all captured chunks into one array
      const allSamples = samplesRef.current;
      if (!allSamples.length) {
        setState("error"); onError?.("No audio captured"); setTimeout(()=>setState("idle"),3000); return;
      }
      const total = allSamples.reduce((s,c)=>s+c.length,0);
      const merged = new Float32Array(total);
      let off=0; for(const c of allSamples){merged.set(c,off);off+=c.length;}

      // Build WAV and send to Vosk (no ffmpeg needed — already 16kHz mono PCM)
      const wav = buildWav(merged, 16000);
      const fd = new FormData(); fd.append("audio", wav, "rec.wav");
      const {data} = await axios.post(`${API}/voice_search`, fd, {timeout:30000});
      if (data.success && data.transcript) { setState("idle"); onResult(data.transcript); }
      else { setState("error"); onError?.(data.error||"No speech detected — speak clearly and try again"); setTimeout(()=>setState("idle"),4000); }
    } catch(err) {
      const detail = err?.response?.data?.detail || err.message || "";
      setState("error");
      onError?.(err?.response?.status===503 ? "Vosk model not loaded" : detail.slice(0,60) || "Voice failed");
      setTimeout(()=>setState("idle"),5000);
    }
  }, [state]);

  return { state, ok, start, stop };
}

/* ══════════════════════════════════════════════════════════════════════════ */
export default function App() {
  const [view,        setView]        = useState("timeline");
  const [images,      setImages]      = useState([]);
  const [faces,       setFaces]       = useState([]);
  const [albums,      setAlbums]      = useState([]);
  const [dupes,       setDupes]       = useState([]);
  const [trash,       setTrash]       = useState([]);
  const [stats,       setStats]       = useState(null);
  const [loading,     setLoading]     = useState(false);
  const [searchQ,     setSearchQ]     = useState("");
  const [searchMode,  setSearchMode]  = useState("text");
  const [colorPick,   setColorPick]   = useState("blue");
  const [hybridFile,  setHybridFile]  = useState(null);
  const [hybridPrev,  setHybridPrev]  = useState(null);
  const [voiceErr,    setVoiceErr]    = useState("");
  const [reindexMsg,  setReindexMsg]  = useState("");
  const [reindexing,  setReindexing]  = useState(false);

  /* New feature state */
  const [emoSummary,    setEmoSummary]    = useState([]);
  const [emoFilter,     setEmoFilter]     = useState("");        // active emotion filter
  const [onThisDay,     setOnThisDay]     = useState(null);
  const [otdExpanded,   setOtdExpanded]   = useState(false);

  const [batchMode,     setBatchMode]     = useState(false);
  const [selected,      setSelected]      = useState(new Set());
  const [batchTagInput, setBatchTagInput] = useState("");
  const [batchAlbumId,  setBatchAlbumId]  = useState("");

  const [allTags,       setAllTags]       = useState([]);
  const [activeTag,     setActiveTag]     = useState("");

  const [createAlbumOpen, setCreateAlbumOpen] = useState(false);
  const [newAlbumTitle,   setNewAlbumTitle]   = useState("");
  const [newAlbumDesc,    setNewAlbumDesc]    = useState("");
  const [pendingAlbumImages, setPendingAlbumImages] = useState([]); // snapshot of selected at open time

  // Named event creation
  const [createEventOpen, setCreateEventOpen] = useState(false);
  const [newEventTitle,   setNewEventTitle]   = useState("");
  const [newEventType,    setNewEventType]    = useState("Other");
  const [newEventDate,    setNewEventDate]    = useState("");
  const [newEventDesc,    setNewEventDesc]    = useState("");

  // Advanced filters
  const [minPeople,     setMinPeople]     = useState(0);
  const [coPersonIds,   setCoPersonIds]   = useState([]);
  const [showCoModal,   setShowCoModal]   = useState(false);

  // Face similarity
  const [faceSrcFile,   setFaceSrcFile]   = useState(null);
  const [faceSrcPrev,   setFaceSrcPrev]   = useState(null);
  const [showFaceSim,   setShowFaceSim]   = useState(false);

  // People frequency
  const [peopleFreq,    setPeopleFreq]    = useState([]);
  // People suggestions shown when search returns no results
  const [peopleSuggestions, setPeopleSuggestions] = useState([]);

  /* Lightbox */
  const [lb, setLb] = useState(null);
  /* Modals */
  const [personModal, setPersonModal] = useState(null);
  const [albumModal,  setAlbumModal]  = useState(null);

  const uploadRef = useRef(null);
  const voice = useVoice(
    t => { setSearchQ(t); setSearchMode("text"); setVoiceErr(""); runSearch("text",t); },
    setVoiceErr
  );

  /* keyboard */
  useEffect(() => {
    const h = e => {
      if (!lb) return;
      if (e.key==="ArrowRight") shiftLb(1);
      if (e.key==="ArrowLeft")  shiftLb(-1);
      if (e.key==="Escape")     setLb(null);
    };
    window.addEventListener("keydown",h);
    return ()=>window.removeEventListener("keydown",h);
  }, [lb]);

  const shiftLb = useCallback(d=>setLb(p=>p?{...p,idx:(p.idx+d+p.list.length)%p.list.length}:null),[]);
  const openLb  = (list,idx)=>setLb({list,idx});
  const closeLb = ()=>setLb(null);

  /* load */
  useEffect(()=>{
    // Don't reload when view="search" with an active tag/emotion filter —
    // filterByTag and filterByEmotion manage their own data fetch.
    // Calling load() here would immediately overwrite their results with nothing.
    if (view === "search") return;
    load();
  },[view]);
  useEffect(()=>{
    loadEmoSummary(); loadTags(); loadTrashCount(); loadPeopleFreq();

  }, []);
  useEffect(()=>{ if(view==="timeline") loadOnThisDay(); }, [view]);

  const load = async () => {
    setLoading(true);
    try {
      if (view==="timeline")   { const r=await axios.get(`${API}/timeline`);              setImages(r.data.results||[]); }
      if (view==="favorites")  { const r=await axios.get(`${API}/favorites`);             setImages(r.data.results||[]); }
      if (view==="explore")    { const r=await axios.get(`${API}/explore/random?count=24`);setImages(r.data.results||[]); }
      if (view==="faces")      { const r=await axios.get(`${API}/faces`);                 setFaces(r.data.results||[]); }
      if (view==="albums")     { const r=await axios.get(`${API}/albums`);                setAlbums(r.data.results||[]); }
      if (view==="duplicates") { const r=await axios.get(`${API}/duplicates`);            setDupes(r.data.duplicate_groups||[]); }
      if (view==="trash")      { const r=await axios.get(`${API}/trash`);                 setTrash(r.data.results||[]); }
      if (view==="stats")      { const r=await axios.get(`${API}/stats`);                 setStats(r.data); }
    } catch {}
    setSelected(new Set()); setBatchMode(false);
    setLoading(false);
  };

  const loadEmoSummary = async () => {
    try { const r=await axios.get(`${API}/emotions/summary`); setEmoSummary(r.data.emotions||[]); } catch {}
  };
  const loadOnThisDay = async () => {
    try { const r=await axios.get(`${API}/on-this-day`); setOnThisDay(r.data); } catch {}
  };
  const loadTags = async () => {
    try { const r=await axios.get(`${API}/tags`); setAllTags(r.data.tags||[]); } catch {}
  };
  const loadTrashCount = async () => {
    try { const r=await axios.get(`${API}/trash`); setTrash(r.data.results||[]); } catch {}
  };

  const loadPeopleFreq = async () => {
    try { const r=await axios.get(`${API}/people/frequency`); setPeopleFreq(r.data.people||[]); } catch {}
  };

  // Group photo filter
  const filterGroupPhotos = async (n) => {
    setMinPeople(n); setLoading(true); setView("search");
    try { const r=await axios.get(`${API}/group-photos?min_people=${n}`); setImages(r.data.results||[]); }
    catch {} finally { setLoading(false); }
  };

  // Co-occurrence search
  const runCoOccurrence = async (ids) => {
    if (!ids||ids.length<2) return;
    setLoading(true); setView("search"); setShowCoModal(false);
    try { const r=await axios.get(`${API}/co-occurrence?person_ids=${ids.join(",")}`); setImages(r.data.results||[]); }
    catch {} finally { setLoading(false); }
  };

  // Face similarity
  const runFaceSimilarity = async () => {
    if (!faceSrcFile) return;
    setLoading(true); setView("search"); setShowFaceSim(false);
    const fd=new FormData(); fd.append("file", faceSrcFile); fd.append("top_k","30");
    try { const r=await axios.post(`${API}/face-similarity`, fd); setImages(r.data.results||[]); }
    catch(err) { alert("Face search failed: "+(err?.response?.data?.detail||err.message)); }
    finally { setLoading(false); setFaceSrcFile(null); setFaceSrcPrev(null); }
  };

  const runSearch = async (mode=searchMode, q=searchQ) => {
    setEmoFilter(""); setActiveTag(""); setLoading(true); setView("search");
    try {
      const fd=new FormData(); let res;
      if (mode==="text")     { fd.append("query",q);          res=await axios.post(`${API}/search`,fd); }
      if (mode==="describe") { fd.append("description",q);    res=await axios.post(`${API}/search/describe`,fd); }
      if (mode==="image")    { fd.append("file",hybridFile); fd.append("top_k",20); res=await axios.post(`${API}/search/image`,fd); }
      if (mode==="hybrid")   { fd.append("query",q); if(hybridFile)fd.append("file",hybridFile); fd.append("top_k",20); res=await axios.post(`${API}/search/hybrid`,fd); }
      if (mode==="color")    { fd.append("color",colorPick); fd.append("top_k",20); res=await axios.post(`${API}/search/color`,fd); }
      let results = res?.data?.results || [];
      // If text search returned nothing, try person name search as fallback
      // Only for queries that look like person names (not animals/objects/descriptive)
      const _NON_PERSON = new Set([
        "cat","dog","horse","cow","bird","fish","fox","lion","tiger","bear","wolf","pig",
        "rabbit","duck","frog","snake","deer","sheep","goat","chicken","otter","seal",
        "panda","koala","monkey","giraffe","zebra","elephant","whale","shark","eagle",
        "owl","penguin","parrot","bee","ant","spider","butterfly","kitten","puppy",
        "sunset","beach","ocean","mountain","forest","river","flower","tree","sky",
        "snow","rain","night","city","park","road","bridge","building","house","lake",
        "football","soccer","cricket","basketball","tennis","volleyball","swimming",
        "running","cycling","boxing","golf","hockey","baseball","rugby","sport","sports",
        "game","match","player","athlete","stadium","gym","yoga","dance","exercise",
        "car","bus","train","plane","boat","ship","bike","truck",
        "food","pizza","cake","coffee","burger","sushi","pasta","bread",
        "sunrise","rainbow","storm","cloud","wind","landscape","nature","sky",
      ]);
      const _qWords = q.trim().toLowerCase().split(/\s+/);
      const _isPersonQuery = _qWords.length <= 4 &&
        _qWords.some(w => w.length >= 3 && !_NON_PERSON.has(w)) &&
        !_qWords.every(w => ["a","an","the","in","on","at","of","for","with",
          "woman","man","girl","boy","black","white","red","blue","green","yellow",
          "dress","shirt","jacket","hair","wearing","standing"].includes(w));
      if (mode==="text" && results.length===0 && q.trim().length>=2 && _isPersonQuery) {
        try {
          const pr = await axios.get(`${API}/people/search?q=${encodeURIComponent(q.trim())}`);
          if (pr.data.results?.length) {
            results = pr.data.results.flatMap(p => p.results || []);
          }
        } catch {}
      }
      // Show people suggestions if still no results
      if (mode==="text" && results.length===0) {
        const suggestions = res?.data?.people_suggestions || [];
        setPeopleSuggestions(suggestions);
      } else {
        setPeopleSuggestions([]);
      }
      setImages(results);
    } catch { setImages([]); }
    setLoading(false);
  };

  const filterByEmotion = async (emo) => {
    if (emoFilter===emo) { setEmoFilter(""); load(); return; }
    setEmoFilter(emo); setView("search"); setLoading(true);
    try { const fd=new FormData(); fd.append("emotion",emo); const r=await axios.post(`${API}/search/emotion`,fd); setImages(r.data.results||[]); }
    catch { setImages([]); }
    setLoading(false);
  };

  const filterByTag = async (tag) => {
    // Toggle: clicking same tag again clears filter
    if (activeTag === tag) {
      setActiveTag("");
      setView("timeline");
      load();
      return;
    }
    // Set active tag and fetch results directly — do NOT call setView() first.
    // setView triggers a useEffect → load() which overwrites the tag results.
    // Instead: set the tag, switch view, and fetch in one go.
    setActiveTag(tag);
    setView("search");   // use "search" view so load() won't auto-fire for it
    setLoading(true);
    try {
      const r = await axios.get(`${API}/tags/${tag}/images`);
      setImages(r.data.results || []);
    } catch {
      setImages([]);
    }
    setLoading(false);
  };

  const handleUpload = async (files) => {
    setLoading(true);
    for (const f of Array.from(files)) { try { const fd=new FormData(); fd.append("file",f); await axios.post(`${API}/upload`,fd); } catch {} }
    load();
  };

  const doReindex = async () => {
    setReindexing(true); setReindexMsg("Clustering…");
    try { const r=await axios.post(`${API}/recluster`); setReindexMsg(`✓ ${r.data.people} people · ${r.data.albums} albums`); setTimeout(load,500); }
    catch { setReindexMsg("Failed"); }
    setReindexing(false);
  };

  const reprocessEmotions = async () => {
    setReindexing(true); setReindexMsg("Reprocessing emotions…");
    try {
      const r = await axios.post(`${API}/reprocess-emotions`);
      setReindexMsg(`✓ ${r.data.message}`);
      setTimeout(()=>{ loadEmoSummary(); if(view==="stats") load(); }, 3000);
    } catch(err) {
      setReindexMsg("Emotion reprocess failed: " + (err?.response?.data?.detail || err.message));
    }
    setReindexing(false);
  };

  const reprocessColors = async () => {
    setReindexing(true); setReindexMsg("Recomputing colors…");
    try {
      const r = await axios.post(`${API}/reprocess-colors`);
      setReindexMsg(`✓ Colors updated for ${r.data.updated} photos`);
    } catch(err) {
      setReindexMsg("Color reprocess failed: " + (err?.response?.data?.detail || err.message));
    }
    setReindexing(false);
  };

  const reprocessNames = async () => {
    setReindexing(true); setReindexMsg("Auto-naming people…");
    try {
      const r = await axios.post(`${API}/reprocess-names`);
      setReindexMsg(`✓ ${r.data.message}`);
      setTimeout(()=>{ if(view==="faces") load(); }, 1000);
    } catch(err) {
      setReindexMsg("Name detection failed: " + (err?.response?.data?.detail || err.message));
    }
    setReindexing(false);
  };

  const reprocessCaptions = async (forceAll=false) => {
    setReindexing(true); setReindexMsg(forceAll ? "Re-captioning all photos…" : "Captioning photos missing descriptions…");
    try {
      const r = await axios.post(`${API}/recaption${forceAll ? "?force_all=true" : ""}`);
      if (r.data.status === "nothing_to_do") {
        setReindexMsg("✓ All photos already have captions. Use 'Re-caption All' to force.");
      } else {
        setReindexMsg(`✓ ${r.data.message}`);
      }
    } catch(err) {
      setReindexMsg("Caption failed: " + (err?.response?.data?.detail || err.message));
    }
    setReindexing(false);
  };

  const toggleFav = async (id, e) => {
    e?.stopPropagation();
    try { const fd=new FormData(); fd.append("image_id",id); await axios.post(`${API}/toggle_favorite`,fd); if(view==="favorites")load(); } catch {}
  };

  const softDelete = async (id, e) => {
    e?.stopPropagation();
    if (!window.confirm("Move to trash?")) return;
    try {
      const fd=new FormData(); fd.append("image_id",id);
      await axios.post(`${API}/delete_image`,fd);
      setImages(p=>p.filter(i=>i.id!==id));
      loadTrashCount();
    } catch(err) {
      alert("Could not move to trash: " + (err?.response?.data?.detail || err.message));
    }
  };

  const restoreImg = async id => {
    try {
      const fd=new FormData(); fd.append("image_id",id);
      await axios.post(`${API}/restore`,fd);
      setTrash(p=>p.filter(i=>i.id!==id));
    } catch(err) {
      alert("Restore failed: " + (err?.response?.data?.detail || err.message));
    }
  };
  const permDelete = async id => {
    if (!window.confirm("Delete forever? This cannot be undone.")) return;
    try {
      const fd=new FormData(); fd.append("image_id",id);
      await axios.post(`${API}/permanent_delete`,fd);
      setTrash(p=>p.filter(i=>i.id!==id));
    } catch(err) {
      alert("Delete failed: " + (err?.response?.data?.detail || err.message));
    }
  };

  const openPerson = async p => {
    setPersonModal({...p,images:null});
    try { const r=await axios.get(`${API}/people/${p.id}`); setPersonModal(r.data); } catch {}
  };
  const openAlbum = async a => {
    setAlbumModal({...a,images:null});
    try { const r=await axios.get(`${API}/albums/${a.id}`); setAlbumModal(r.data); } catch {}
  };
  const renamePerson = async (id,name) => {
    try { const fd=new FormData(); fd.append("name",name); await axios.post(`${API}/people/${id}`,fd); load(); setPersonModal(p=>p&&{...p,name}); } catch {}
  };

  /* Batch actions */
  const toggleSelect = useCallback((id) => {
    setSelected(prev => { const s=new Set(prev); s.has(id)?s.delete(id):s.add(id); return s; });
  }, []);
  const selectAll = () => setSelected(new Set(images.map(i=>i.id)));
  const clearSel  = () => setSelected(new Set());

  const batchFavorite = async (val) => {
    if (!selected.size) return;
    const fd=new FormData(); fd.append("image_ids",[...selected].join(",")); fd.append("value",val?1:0);
    try { await axios.post(`${API}/batch/favorite`,fd); load(); } catch {}
  };
  const batchDelete = async () => {
    if (!selected.size||!window.confirm(`Trash ${selected.size} photos?`)) return;
    const fd=new FormData(); fd.append("image_ids",[...selected].join(","));
    try { await axios.post(`${API}/batch/delete`,fd); load(); } catch {}
  };
  const batchTag = async () => {
    if (!selected.size||!batchTagInput.trim()) return;
    const fd=new FormData(); fd.append("image_ids",[...selected].join(",")); fd.append("tag",batchTagInput.trim());
    try { await axios.post(`${API}/batch/tag`,fd); setBatchTagInput(""); loadTags(); } catch {}
  };
  const batchAddAlbum = async () => {
    if (!selected.size||!batchAlbumId) return;
    const fd=new FormData(); fd.append("image_ids",[...selected].join(",")); fd.append("album_id",batchAlbumId);
    try { await axios.post(`${API}/batch/album`,fd); setBatchAlbumId(""); load(); } catch {}
  };

  const createAlbum = async (imageIds=[]) => {
    if (!newAlbumTitle.trim()) return;
    const fd=new FormData();
    fd.append("title", newAlbumTitle);
    fd.append("description", newAlbumDesc);
    // Priority: explicit imageIds arg → pendingAlbumImages snapshot → current selected
    const ids = imageIds.length ? imageIds
              : pendingAlbumImages.length ? pendingAlbumImages
              : [...selected];
    console.log("Creating album with", ids.length, "images:", ids);
    if (ids.length) fd.append("image_ids", ids.join(","));
    try {
      const r = await axios.post(`${API}/albums/create`, fd);
      setCreateAlbumOpen(false); setNewAlbumTitle(""); setNewAlbumDesc("");
      const addedCount = ids.length;
      setPendingAlbumImages([]);
      setSelected(new Set()); setBatchMode(false);
      setView("albums");
      await load();
      // Open the newly created album immediately so user can see it
      if (r.data?.id) {
        const newAlbum = {id: r.data.id, title: r.data.title, type:"manual",
                         count: addedCount, images: null};
        openAlbum(newAlbum);
      }
      return r.data;
    } catch(err) { alert("Failed to create album: " + (err?.response?.data?.detail || err.message)); }
  };
  const deleteAlbum = async id => {
    try {
      await axios.delete(`${API}/albums/${id}/delete`);
      setAlbumModal(null);
      load();
    } catch(err) { alert("Delete failed: " + (err?.response?.data?.detail || err.message)); }
  };

  const renameAlbum = async (id, title, description) => {
    const fd=new FormData(); fd.append("title", title);
    if (description) fd.append("description", description);
    try {
      await axios.post(`${API}/albums/${id}/rename`, fd);
      load();
      setAlbumModal(prev => prev && prev.id===id ? {...prev, title, description: description||prev.description} : prev);
    } catch(err) { alert("Rename failed: " + (err?.response?.data?.detail || err.message)); }
  };

  const EVENT_TYPES = ["Birthday","Trip","Vacation","Wedding","Anniversary","Graduation","Party","Holiday","Family","Work","Other"];

  const createNamedEvent = async () => {
    if (!newEventTitle.trim()) return;
    const fd=new FormData();
    fd.append("title", newEventTitle.trim());
    fd.append("event_type", newEventType);
    fd.append("description", newEventDesc.trim());
    fd.append("date_str", newEventDate);
    try {
      await axios.post(`${API}/events/create`, fd);
      setCreateEventOpen(false);
      setNewEventTitle(""); setNewEventType("Other"); setNewEventDate(""); setNewEventDesc("");
      load();
    } catch(err) { alert("Could not create event: " + (err?.response?.data?.detail || err.message)); }
  };

  const addTagToImage = async (imgId, tag) => {
    const fd=new FormData(); fd.append("tag",tag);
    try {
      const r = await axios.post(`${API}/photo/${imgId}/tags/add`,fd);
      const newTags = r.data.tags || [];
      // Update the images list in place so lightbox reflects latest tags
      setImages(prev => prev.map(i => i.id===imgId ? {...i, user_tags: newTags} : i));
      loadTags();
      return newTags;
    } catch { return null; }
  };
  const removeTagFromImage = async (imgId, tag) => {
    const fd=new FormData(); fd.append("tag",tag);
    try {
      const r = await axios.post(`${API}/photo/${imgId}/tags/remove`,fd);
      const newTags = r.data.tags || [];
      setImages(prev => prev.map(i => i.id===imgId ? {...i, user_tags: newTags} : i));
      loadTags();
      return newTags;
    } catch { return null; }
  };

  const isGridView = ["timeline","search","favorites","explore"].includes(view);

  const NAV = [
    { id:"timeline",   icon:<Clock size={15}/>,     label:"Timeline"   },
    { id:"albums",     icon:<BookOpen size={15}/>,  label:"Albums"     },
    { id:"faces",      icon:<Users size={15}/>,     label:"People"     },
    { id:"favorites",  icon:<Heart size={15}/>,     label:"Favorites"  },
    { id:"duplicates", icon:<Copy size={15}/>,      label:"Duplicates" },
    { id:"explore",    icon:<Shuffle size={15}/>,   label:"Explore"    },
    { id:"stats",      icon:<BarChart3 size={15}/>, label:"Statistics" },
  ];
  const MODES = [
    { id:"text",     Icon:Search,   label:"Search"   },
    { id:"describe", Icon:FileText, label:"Describe" },
    { id:"image",    Icon:ImgIcon,  label:"By Image" },
    { id:"hybrid",   Icon:GitMerge, label:"Hybrid"   },
    { id:"color",    Icon:Palette,  label:"By Color" },
  ];
  const COLORS = ["red","orange","yellow","green","blue","purple","pink","white","black","gray","brown"];

  return (
    <>
      <CSS />
      <div className="shell">

        {/* ── SIDEBAR ─────────────────────────────────────────────────── */}
        <nav className="sidebar">
          <div className="brand">
            <div className="brand-mark"><Aperture size={16} strokeWidth={1.5}/></div>
            <div>
              <p className="brand-name">LUMINA</p>
              <p className="brand-sub">AI Gallery</p>
            </div>
          </div>

          <div className="nav-list">
            {NAV.map(n=>(
              <button key={n.id} className={`nav-btn ${view===n.id?"nav-btn--on":""}`} onClick={()=>setView(n.id)}>
                <span className="nav-ic">{n.icon}</span>
                <span>{n.label}</span>
                {view===n.id && <span className="nav-pip"/>}
              </button>
            ))}
          </div>

          <div className="sidebar-sep"/>

          {/* Advanced Filters */}
          <p className="sidebar-section-label"><SlidersHorizontal size={10}/> FILTERS</p>
          <button className="nav-btn" style={{fontSize:11}} onClick={()=>filterGroupPhotos(2)}>
            <span className="nav-ic"><Users size={14}/></span><span>2+ People</span>
          </button>
          <button className="nav-btn" style={{fontSize:11}} onClick={()=>filterGroupPhotos(3)}>
            <span className="nav-ic"><Users size={14}/></span><span>3+ People</span>
          </button>
          <button className="nav-btn" style={{fontSize:11}} onClick={()=>filterGroupPhotos(5)}>
            <span className="nav-ic"><Users size={14}/></span><span>5+ People</span>
          </button>
          <button className="nav-btn" style={{fontSize:11}} onClick={()=>setShowCoModal(true)}>
            <span className="nav-ic"><GitMerge size={14}/></span><span>Co-Appear</span>
          </button>
          <button className="nav-btn" style={{fontSize:11}} onClick={()=>setShowFaceSim(true)}>
            <span className="nav-ic"><UserCheck size={14}/></span><span>Face Search</span>
          </button>

          <div className="sidebar-sep"/>

          <button className={`nav-btn nav-btn--trash ${view==="trash"?"nav-btn--on":""}`} onClick={()=>setView("trash")}>
            <span className="nav-ic"><Trash size={15}/></span>
            <span>Trash</span>
            {trash.length>0 && <span className="badge">{trash.length}</span>}
          </button>

          {/* Tags sidebar section */}
          {allTags.length>0 && (
            <>
              <div className="sidebar-sep"/>
              <p className="sidebar-section-label"><Hash size={10}/> TAGS</p>
              <div className="tag-nav-list">
                {allTags.slice(0,8).map(t=>(
                  <button key={t.tag} className={`tag-nav-btn ${activeTag===t.tag?"tag-nav-btn--on":""}`} onClick={()=>filterByTag(t.tag)}>
                    <span>#{t.tag}</span><span className="tag-nav-count">{t.count}</span>
                  </button>
                ))}
              </div>
            </>
          )}

          <div className="sidebar-sep"/>
          <div className="sidebar-foot">
            <button className="reindex-btn reindex-btn--primary" onClick={doReindex} disabled={reindexing}>
              <Zap size={13} className={reindexing?"spin":""}/>{reindexing?"Indexing…":"Re-index AI"}
            </button>
            <ToolsAccordion reindexing={reindexing}
              onEmotions={reprocessEmotions}
              onColors={reprocessColors}
              onNames={reprocessNames}
              onCaptions={()=>reprocessCaptions(false)}
              onRecaptionAll={()=>reprocessCaptions(true)}
              onCleanup={async()=>{ if(!window.confirm("Delete all empty albums?")) return; const r=await axios.delete(`${API}/albums/empty/cleanup`); setReindexMsg(`✓ Deleted ${r.data.deleted} empty albums`); load(); }}
            />
            {reindexMsg && <p className="reindex-msg">{reindexMsg}</p>}
            <div className="offline-pill"><span className="dot-green"/><span>Fully Offline</span></div>
            {voiceErr && voiceErr.includes("setup") && (
              <div style={{background:"rgba(245,158,11,.08)",border:"1px solid rgba(245,158,11,.2)",borderRadius:7,padding:"8px 10px",fontSize:10,color:"rgba(245,158,11,.9)",lineHeight:1.5}}>
                <strong>Voice Setup:</strong><br/>
                1. pip install vosk<br/>
                2. Download model from<br/>
                <span style={{color:"#60a5fa",wordBreak:"break-all"}}>alphacephei.com/vosk/models</span><br/>
                3. Extract to:<br/>
                <code style={{fontSize:9,color:"#aaa"}}>models/vosk-model-small-en-us</code>
              </div>
            )}
          </div>
        </nav>

        {/* ── MAIN ────────────────────────────────────────────────────── */}
        <div className="main">

          {/* TOPBAR */}
          <header className="topbar">
            <div className="mode-strip">
              {MODES.map(({id,Icon,label})=>(
                <button key={id} className={`mode-btn ${searchMode===id?"mode-btn--on":""}`} onClick={()=>setSearchMode(id)}>
                  <Icon size={11}/> {label}
                </button>
              ))}
              {voice.state!=="idle" && (
                <span className={`voice-chip voice-chip--${voice.state}`}>
                  {voice.state==="rec"        && <><MicOff size={10}/> Recording</>}
                  {voice.state==="processing" && <><Loader2 size={10} className="spin"/> Transcribing…</>}
                  {voice.state==="error"      && <span title={voiceErr}>{voiceErr.length>38?voiceErr.slice(0,38)+"…":voiceErr}</span>}
                </span>
              )}
            </div>
            {/* ── Quick emoji search chips ───────────────────────────────── */}
            <div className="emoji-quick-bar">
              <span className="emoji-quick-label">Quick:</span>
              {[["😊","happy"],["😢","sad"],["😠","angry"],["😲","surprised"],["😨","fearful"],
                ["🐶","dog"],["🏖️","beach"],["🌅","sunset"],["🎉","party"],["✈️","travel"],
                ["❄️","snow"],["🌸","flowers"],["🍕","food"],["🏋️","gym"],["🌊","ocean"]
              ].map(([emoji,label])=>(
                <button key={emoji} className="emoji-quick-btn" title={label}
                  onClick={()=>{ setSearchQ(emoji); setSearchMode("text"); runSearch("text", emoji); }}>
                  {emoji}
                </button>
              ))}
            </div>
            <div className="search-bar">
              {(searchMode==="text"||searchMode==="describe"||searchMode==="hybrid") && (
                <div className="search-wrap">
                  <Search size={14} className="search-ic"/>
                  <input className="search-in" value={searchQ}
                    onChange={e => setSearchQ(e.target.value)}
                    onKeyDown={e => {
                      if (e.key === "Enter") {
                        // Pass e.target.value directly — React state (searchQ) may not
                        // have updated yet when the keydown fires, causing the first
                        // Enter to search with the previous value (stale closure bug).
                        const val = e.target.value;
                        setSearchQ(val);
                        runSearch(searchMode, val);
                      }
                    }}
                    placeholder={searchMode==="describe"?"Describe what's in the photo…":searchMode==="hybrid"?"Text + optional image…":"Search by scene, object, person, text…"}
                  />
                  {voice.ok && (
                    <button className={`mic-btn ${voice.state==="rec"?"mic-btn--rec":voice.state==="processing"?"mic-btn--proc":""}`}
                      onClick={voice.state==="rec"?voice.stop:voice.start}>
                      {voice.state==="rec"?<MicOff size={14}/>:voice.state==="processing"?<Loader2 size={14} className="spin"/>:<Mic size={14}/>}
                    </button>
                  )}
                </div>
              )}
              {(searchMode==="image"||searchMode==="hybrid") && (
                <label className="img-search-label">
                  {hybridPrev?<img src={hybridPrev} className="img-search-thumb" alt=""/>:<ImgIcon size={13}/>}
                  <span>{hybridPrev?"Ready":"Upload ref"}</span>
                  <input type="file" accept="image/*" style={{display:"none"}} onChange={e=>{const f=e.target.files?.[0];if(f){setHybridFile(f);setHybridPrev(URL.createObjectURL(f))}}}/>
                </label>
              )}
              {searchMode==="color" && (
                <div className="color-strip">
                  {COLORS.map(c=>(
                    <button key={c} title={c} className={`clr-dot ${colorPick===c?"clr-dot--on":""}`}
                      style={{background:c==="white"?"#e5e5e5":c==="gray"?"#888":c}}
                      onClick={()=>setColorPick(c)}/>
                  ))}
                </div>
              )}
              <button className="btn-go" onClick={()=>runSearch(searchMode, searchQ)}>Search</button>
              {isGridView && (
                <button className={`btn-batch ${batchMode?"btn-batch--on":""}`} onClick={()=>{setBatchMode(p=>!p);clearSel();}}>
                  <CheckSquare size={13}/> {batchMode?"Exit Select":"Select"}
                </button>
              )}
<button className="btn-upload" onClick={()=>uploadRef.current?.click()}>
                <Upload size={13}/> Upload
              </button>
              <input ref={uploadRef} type="file" multiple accept="image/*" style={{display:"none"}} onChange={e=>handleUpload(e.target.files)}/>
            </div>

            {/* Emotion filter bar */}
            {isGridView && emoSummary.length>0 && (
              <div className="emo-filter-bar">
                <span className="emo-filter-label">Filter:</span>
                {emoSummary.filter(e=>e.emotion&&e.emotion!=="neutral").map(e=>(
                  <button key={e.emotion}
                    className={`emo-filter-btn ${emoFilter===e.emotion?"emo-filter-btn--on":""}`}
                    style={emoFilter===e.emotion?{background:EMO_COLORS[e.emotion]+"22",borderColor:EMO_COLORS[e.emotion]+"66",color:EMO_COLORS[e.emotion]}:{}}
                    onClick={()=>filterByEmotion(e.emotion)}>
                    {EMO[e.emotion]||"😐"} {e.emotion} <span className="emo-filter-count">{e.count}</span>
                  </button>
                ))}
                {(emoFilter||activeTag) && (
                  <button className="emo-filter-clear" onClick={()=>{setEmoFilter("");setActiveTag("");load();}}>
                    <X size={10}/> Clear
                  </button>
                )}
              </div>
            )}
          </header>

          {/* BODY */}
          <div className="body">
            <AnimatePresence mode="wait">
              {loading ? (
                <Fade key="load">
                  <div className="loader-wrap"><div className="loader-ring"/><p className="loader-text">Loading…</p></div>
                </Fade>

              ) : isGridView ? (
                <Fade key={view+emoFilter+activeTag}>
                  {/* On This Day banner */}
                  {view==="timeline" && onThisDay?.total>0 && (
                    <OnThisDayBanner data={onThisDay} expanded={otdExpanded} onToggle={()=>setOtdExpanded(p=>!p)} onOpen={openLb}/>
                  )}

                  <div className="page-head">
                    <h1 className="page-title">
                      {activeTag ? `#${activeTag}` :
                       emoFilter ? `${EMO[emoFilter]} ${emoFilter}` :
                       view==="search" ? `Results for "${searchQ}"` :
                       view==="favorites" ? "Favorites" :
                       view==="explore" ? "Explore" : "Timeline"}
                    </h1>
                    {images.length>0 && <span className="page-count">{images.length} photos</span>}
                    {batchMode && selected.size>0 && <span className="sel-count">{selected.size} selected</span>}
                    {view==="explore" && <button className="btn-sm" style={{marginLeft:"auto"}} onClick={load}><Shuffle size={11}/> Shuffle</button>}
                  </div>

                  {images.length===0
                    ? (view==="search" && peopleSuggestions.length>0 ? (
                        <div>
                          <Empty icon={ImageOff} msg="No photos matched" sub="But these people are in your gallery — click to view their photos"/>
                          <div style={{marginTop:16}}>
                            <p style={{fontSize:11,color:"var(--mu)",marginBottom:10,fontFamily:"var(--mono)",letterSpacing:".08em"}}>PEOPLE IN YOUR GALLERY:</p>
                            <div className="people-grid">
                              {peopleSuggestions.map(p=>(
                                <motion.div key={p.id} className="person-card" whileHover={{y:-3}}
                                  onClick={async()=>{
                                    setLoading(true); setView("search"); setPeopleSuggestions([]);
                                    try { const r=await axios.get(`${API}/people/${p.id}`); setImages(r.data.results||r.data.images||[]); }
                                    catch {} finally { setLoading(false); }
                                  }}>
                                  <div className="person-avatar">
                                    {p.cover?<img src={imgUrl(p.cover)} alt="" onError={e=>e.target.src=BLANK}/>:<Users size={26} strokeWidth={1} color="#444"/>}
                                  </div>
                                  <p className="person-name">{p.name}</p>
                                  <p className="person-count" style={{color:"var(--ac2)",fontSize:9}}>click to rename</p>
                                </motion.div>
                              ))}
                            </div>
                            <p style={{fontSize:11,color:"#555",marginTop:14,padding:"10px 14px",background:"var(--s2)",borderRadius:8,border:"1px solid var(--br)"}}>
                              💡 <strong>Tip:</strong> Click a person above → click ✏ Rename → type their real name → search will work instantly
                            </p>
                          </div>
                        </div>
                      ) : <Empty icon={ImageOff} msg={view==="search"?"No results":"No photos yet"} sub={view!=="search"?"Upload photos to get started":"Try different keywords"} onClick={view!=="search"?()=>uploadRef.current?.click():null}/>)
                    : <ImgGrid list={images} onOpen={i=>openLb(images,i)} onFav={toggleFav} onDel={softDelete} batchMode={batchMode} selected={selected} onToggle={toggleSelect}/>
                  }
                </Fade>

              ) : view==="faces" ? (
                <Fade key="faces">
                  <PageHead title="People" count={faces.length}/>
                  {faces.length===0
                    ? <Empty icon={Users} msg="No people detected" sub="Run Re-index to detect faces"/>
                    : <div className="people-grid">{faces.map(p=><PersonCard key={p.id} p={p} onClick={()=>openPerson(p)}/>)}</div>
                  }
                </Fade>

              ) : view==="albums" ? (
                <Fade key="albums">
                  <div className="page-head">
                    <h1 className="page-title">Albums</h1>
                    {albums.length>0 && <span className="page-count">{albums.length}</span>}
                    <button className="btn-sm btn-sm--accent" style={{marginLeft:"auto"}} onClick={()=>{ setPendingAlbumImages([]); setCreateAlbumOpen(true); }}>
                      <Plus size={12}/> New Album
                    </button>
                    <button className="btn-sm" style={{marginLeft:6,background:"var(--s3)",border:"1px solid var(--br)"}} onClick={()=>setCreateEventOpen(true)}>🎉 New Event</button>
                  </div>
                  {albums.length===0
                    ? <Empty icon={BookImage} msg="No albums" sub="Run Re-index to auto-generate or create manually"/>
                    : <div className="albums-grid">{albums.map(a=><AlbumCard key={a.id} a={a} onClick={()=>openAlbum(a)} onDelete={a.type==="manual"?()=>deleteAlbum(a.id):null}/>)}</div>
                  }
                </Fade>

              ) : view==="duplicates" ? (
                <Fade key="dupes">
                  <PageHead title="Duplicate Groups" count={dupes.length} unit="groups"/>
                  {dupes.length===0
                    ? <Empty icon={Copy} msg="No duplicates" sub="All images are unique"/>
                    : <div className="dupe-list">
                        {dupes.map((g,gi)=>(
                          <div key={gi} className="dupe-group">
                            <div className="dupe-head">
                              <span className="tag-sm">Group {gi+1}</span>
                              <span className="tag-sm tag-sm--accent">{g.count} duplicates</span>
                              {g.total_size&&<span className="tag-sm">{(g.total_size/1024/1024).toFixed(1)} MB</span>}
                            </div>
                            <ImgGrid list={g.images||[]} compact onOpen={i=>openLb(g.images,i)} onFav={toggleFav} onDel={softDelete}/>
                          </div>
                        ))}
                      </div>
                  }
                </Fade>

              ) : view==="trash" ? (
                <Fade key="trash">
                  <PageHead title="Trash" count={trash.length}/>
                  {trash.length===0 ? <Empty icon={Trash} msg="Trash is empty"/> : (
                    <div className="img-grid">
                      {trash.map(img=>(
                        <div key={img.id} className="trash-card">
                          <img src={imgUrl(img.filename)} alt="" className="trash-img" onError={e=>e.target.src=BLANK}/>
                          <div className="trash-overlay">
                            <button className="trash-btn trash-btn--restore" onClick={()=>restoreImg(img.id)}><RotateCcw size={11}/> Restore</button>
                            <button className="trash-btn trash-btn--del" onClick={()=>permDelete(img.id)}><AlertTriangle size={11}/> Delete Forever</button>
                          </div>
                          {img.trashed_at&&<span className="trash-date">{new Date(img.trashed_at).toLocaleDateString()}</span>}
                        </div>
                      ))}
                    </div>
                  )}
                </Fade>

              ) : view==="stats" ? (
                <Fade key="stats"><StatsPage stats={stats} peopleFreq={peopleFreq}/></Fade>
              ) : null}
            </AnimatePresence>
          </div>

          {/* BATCH ACTION BAR */}
          <AnimatePresence>
            {batchMode && (
              <motion.div className="batch-bar"
                initial={{y:60,opacity:0}} animate={{y:0,opacity:1}} exit={{y:60,opacity:0}}>
                <div className="batch-bar-left">
                  <span className="batch-count"><CheckCheck size={14}/> {selected.size} selected</span>
                  <button className="batch-sm" onClick={selectAll}>All</button>
                  <button className="batch-sm" onClick={clearSel}>None</button>
                </div>
                <div className="batch-bar-actions">
                  <button className="batch-action-btn batch-action-btn--fav" onClick={()=>batchFavorite(true)} title="Favorite all"><Heart size={13}/> Favorite</button>
                  <button className="batch-action-btn batch-action-btn--del" onClick={batchDelete} title="Trash all"><Trash2 size={13}/> Trash</button>
                  <div className="batch-input-group">
                    <input className="batch-input" placeholder="Add tag…" value={batchTagInput} onChange={e=>setBatchTagInput(e.target.value)} onKeyDown={e=>e.key==="Enter"&&batchTag()}/>
                    <button className="batch-action-btn" onClick={batchTag}><Tag size={12}/></button>
                  </div>
                  <div className="batch-input-group">
                    <select className="batch-select" value={batchAlbumId} onChange={e=>setBatchAlbumId(e.target.value)}>
                      <option value="">Add to album…</option>
                      {albums.map(a=><option key={a.id} value={a.id}>{a.title}</option>)}
                    </select>
                    <button className="batch-action-btn" onClick={batchAddAlbum} title="Add to existing album"><Layers size={12}/></button>
                  </div>
                  <button className="batch-action-btn batch-action-btn--album"
                    onClick={()=>{
                      setPendingAlbumImages([...selected]); // snapshot NOW before modal opens
                      setCreateAlbumOpen(true);
                    }}
                    title={`Create new album with ${selected.size} selected photos`}
                    disabled={selected.size===0}>
                    <FolderPlus size={12}/> New Album
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* LIGHTBOX */}
      <AnimatePresence>
        {lb && <Lightbox list={lb.list} idx={lb.idx} onClose={closeLb} onShift={shiftLb} onFav={toggleFav} onDel={(id,e)=>{softDelete(id,e);closeLb();}} onAddTag={addTagToImage} onRemoveTag={removeTagFromImage} albums={albums}/>}
      </AnimatePresence>

      {/* PERSON MODAL */}
      <AnimatePresence>
        {personModal && <PersonModal data={personModal} onClose={()=>setPersonModal(null)} onRename={renamePerson} onOpen={(list,i)=>{setPersonModal(null);setTimeout(()=>openLb(list,i),80);}}/>}
      </AnimatePresence>

      {/* ALBUM MODAL */}
      <AnimatePresence>
        {albumModal && <AlbumModal data={albumModal} onClose={()=>setAlbumModal(null)} onRename={renameAlbum} onDelete={deleteAlbum} onOpen={(list,i)=>{setAlbumModal(null);setTimeout(()=>openLb(list,i),80);}}/>}
      </AnimatePresence>

      {/* CREATE ALBUM MODAL */}
      <AnimatePresence>
        {createAlbumOpen && (
          <ModalWrap onClose={()=>setCreateAlbumOpen(false)}>
            <div className="modal-head">
              <h2 className="modal-title">New Album</h2>
              <button className="lb-hbtn lb-close" onClick={()=>setCreateAlbumOpen(false)}><X size={17}/></button>
            </div>
            {pendingAlbumImages.length>0 && (
              <div style={{background:"rgba(96,165,250,.08)",border:"1px solid rgba(96,165,250,.2)",borderRadius:8,padding:"8px 12px",marginBottom:12,fontSize:12,color:"rgba(96,165,250,.9)"}}>
                📸 <strong>{pendingAlbumImages.length} selected photos</strong> will be added to this album
              </div>
            )}
            <div className="create-album-form">
              <input className="form-input" placeholder="Album title…" value={newAlbumTitle}
                onChange={e=>setNewAlbumTitle(e.target.value)}
                onKeyDown={e=>e.key==="Enter"&&newAlbumTitle.trim()&&createAlbum()}
                autoFocus/>
              <textarea className="form-textarea" placeholder="Description (optional)…"
                value={newAlbumDesc} onChange={e=>setNewAlbumDesc(e.target.value)} rows={2}/>
              <div style={{display:"flex",gap:8}}>
                <button className="btn-sm btn-sm--primary" style={{flex:1}} onClick={()=>createAlbum()} disabled={!newAlbumTitle.trim()}>
                  <FolderPlus size={13}/> Create Album
                </button>
                <button className="btn-sm" onClick={()=>setCreateAlbumOpen(false)}>Cancel</button>
              </div>
            </div>
          </ModalWrap>
        )}
      </AnimatePresence>

      {/* CREATE NAMED EVENT MODAL */}
      <AnimatePresence>
        {createEventOpen && (
          <ModalWrap onClose={()=>setCreateEventOpen(false)}>
            <div className="modal-head">
              <h2 className="modal-title">🎉 New Event</h2>
              <button className="lb-hbtn lb-close" onClick={()=>setCreateEventOpen(false)}><X size={17}/></button>
            </div>
            <div className="create-album-form">
              <label style={{fontSize:11,color:"var(--mu)",marginBottom:4}}>EVENT TYPE</label>
              <div style={{display:"flex",flexWrap:"wrap",gap:6,marginBottom:12}}>
                {["Birthday","Trip","Vacation","Wedding","Anniversary","Graduation","Party","Holiday","Family","Work","Other"].map(t=>(
                  <button key={t}
                    className={"btn-sm" + (newEventType===t?" btn-sm--primary":"")}
                    style={{fontSize:11,padding:"4px 10px"}}
                    onClick={()=>setNewEventType(t)}>
                    {t==="Birthday"?"🎂":t==="Trip"?"✈️":t==="Vacation"?"🏖️":t==="Wedding"?"💍":t==="Anniversary"?"💕":t==="Graduation"?"🎓":t==="Party"?"🎊":t==="Holiday"?"🎄":t==="Family"?"👨‍👩‍👧":t==="Work"?"💼":"📅"} {t}
                  </button>
                ))}
              </div>
              <input className="form-input" placeholder="Event name (e.g. Mom's 60th Birthday)…" value={newEventTitle} onChange={e=>setNewEventTitle(e.target.value)} autoFocus/>
              <input className="form-input" type="date" value={newEventDate} onChange={e=>setNewEventDate(e.target.value)} style={{marginTop:8}}/>
              <textarea className="form-textarea" placeholder="Notes (optional)…" value={newEventDesc} onChange={e=>setNewEventDesc(e.target.value)} rows={2} style={{marginTop:8}}/>
              <button className="btn-sm btn-sm--primary" onClick={createNamedEvent} disabled={!newEventTitle.trim()} style={{marginTop:4}}>
                🎉 Create Event
              </button>
            </div>
          </ModalWrap>
        )}
      </AnimatePresence>

      {/* CO-OCCURRENCE MODAL */}
      <AnimatePresence>
        {showCoModal && (
          <ModalWrap onClose={()=>setShowCoModal(false)}>
            <div className="modal-head">
              <h2 className="modal-title"><GitMerge size={16} style={{marginRight:6,verticalAlign:"middle"}}/>Co-Appearance Search</h2>
              <button className="lb-hbtn lb-close" onClick={()=>setShowCoModal(false)}><X size={17}/></button>
            </div>
            <CoOccurrencePanel faces={faces} onSearch={runCoOccurrence}/>
          </ModalWrap>
        )}
      </AnimatePresence>

      {/* FACE SIMILARITY MODAL */}
      <AnimatePresence>
        {showFaceSim && (
          <ModalWrap onClose={()=>setShowFaceSim(false)}>
            <div className="modal-head">
              <h2 className="modal-title"><UserCheck size={16} style={{marginRight:6,verticalAlign:"middle"}}/>Face Similarity Search</h2>
              <button className="lb-hbtn lb-close" onClick={()=>setShowFaceSim(false)}><X size={17}/></button>
            </div>
            <div className="create-album-form">
              <p style={{fontSize:12,color:"var(--mu)",marginBottom:12}}>Upload a face crop or photo — find all photos with the same person.</p>
              {faceSrcPrev && <img src={faceSrcPrev} style={{width:120,height:120,objectFit:"cover",borderRadius:8,marginBottom:12,border:"2px solid var(--ac)"}} alt="face"/>}
              <input type="file" accept="image/*" style={{display:"none"}} id="face-sim-input"
                onChange={e=>{
                  const f=e.target.files[0]; if(!f) return;
                  setFaceSrcFile(f);
                  const r=new FileReader(); r.onload=ev=>setFaceSrcPrev(ev.target.result); r.readAsDataURL(f);
                }}/>
              <button className="btn-sm" onClick={()=>document.getElementById("face-sim-input").click()}>
                <Camera size={12}/> {faceSrcPrev?"Change Photo":"Upload Face Photo"}
              </button>
              {faceSrcFile && (
                <button className="btn-sm btn-sm--primary" style={{marginTop:10}} onClick={runFaceSimilarity}>
                  <UserCheck size={12}/> Find Similar Faces
                </button>
              )}
            </div>
          </ModalWrap>
        )}
      </AnimatePresence>
    </>
  );
}

/* ══════════════════════════════════════════════════════════════════════════ */
/* ON THIS DAY BANNER                                                          */
/* ══════════════════════════════════════════════════════════════════════════ */
function OnThisDayBanner({ data, expanded, onToggle, onOpen }) {
  return (
    <div className="otd-banner">
      <div className="otd-header" onClick={onToggle}>
        <div className="otd-left">
          <Calendar size={15} color="#f59e0b"/>
          <div>
            <span className="otd-title">On This Day — {data.date}</span>
            <span className="otd-sub">{data.total} photo{data.total!==1?"s":""} from {data.years?.length} year{data.years?.length!==1?"s":""} ago</span>
          </div>
        </div>
        <button className="otd-toggle">{expanded?<ChevronUp size={14}/>:<ChevronDown size={14}/>}</button>
      </div>
      <AnimatePresence>
        {expanded && (
          <motion.div initial={{height:0,opacity:0}} animate={{height:"auto",opacity:1}} exit={{height:0,opacity:0}}>
            {data.years?.map(yr=>(
              <div key={yr.year} className="otd-year">
                <p className="otd-year-label">{yr.year} — {yr.count} photo{yr.count!==1?"s":""}</p>
                <div className="otd-strip">
                  {yr.images.map((img,i)=>(
                    <div key={img.id} className="otd-thumb" onClick={()=>onOpen(yr.images,i)}>
                      <img src={imgUrl(img.filename)} alt="" onError={e=>e.target.src=BLANK}/>
                      {img.caption_short && <div className="otd-thumb-caption">{img.caption_short}</div>}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}


/* ══════════════════════════════════════════════════════════════════════════ */
/* IMAGE GRID                                                                  */
/* ══════════════════════════════════════════════════════════════════════════ */
function ImgGrid({ list, onOpen, onFav, onDel, compact, batchMode, selected, onToggle }) {
  return (
    <div className={`img-grid${compact?" img-grid--sm":""}`}>
      {list.map((img,i)=>(
        <PhotoCard key={img.id??i} img={img} onOpen={()=>onOpen(i)} onFav={onFav} onDel={onDel}
          batchMode={batchMode} isSelected={selected?.has(img.id)} onToggle={onToggle}/>
      ))}
    </div>
  );
}

function PhotoCard({ img, onOpen, onFav, onDel, batchMode, isSelected, onToggle }) {
  const q=qInfo(img.quality_level); const emo=EMO[img.dominant_emotion]||"";
  return (
    <motion.div
      className={`photo-card ${isSelected?"photo-card--sel":""}`}
      whileHover={{y:-4, scale:1.02}}
      whileTap={{scale:0.97}}
      transition={{type:"spring", stiffness:380, damping:28}}
      onClick={batchMode?()=>onToggle(img.id):onOpen}
    >
      <img src={imgUrl(img.filename)||BLANK} alt="" className="photo-img" loading="lazy"
        onError={e=>{e.target.onerror=null;e.target.src=BLANK;}}/>

      {/* Top chips */}
      {img.quality_level&&img.quality_level!=="Processing..."&&(
        <span className="chip chip--q" style={{color:q.c,background:q.bg,borderColor:q.c+"44"}}>{img.quality_level[0]}</span>
      )}
      {emo&&img.dominant_emotion!=="neutral"&&<span className="chip chip--emo">{emo}</span>}
      {img.is_favorite&&<span className="fav-dot">♥</span>}

      {batchMode ? (
        <motion.div className="batch-check" initial={{scale:0.5}} animate={{scale:1}}>
          {isSelected
            ? <CheckSquare size={20} color="#7c6af7" style={{filter:"drop-shadow(0 0 6px #7c6af7aa)"}}/>
            : <Square size={20} color="rgba(255,255,255,.4)"/>
          }
        </motion.div>
      ) : (
        <div className="photo-hover">
          <div className="photo-actions">
            <motion.button whileHover={{scale:1.15}} whileTap={{scale:.9}}
              className="ph-btn ph-btn--fav" onClick={e=>{e.stopPropagation();onFav(img.id,e);}}>
              <Heart size={12} fill={img.is_favorite?"#fff":"none"}/>
            </motion.button>
            <motion.button whileHover={{scale:1.15}} whileTap={{scale:.9}}
              className="ph-btn ph-btn--del" onClick={e=>{e.stopPropagation();onDel(img.id,e);}}>
              <Trash2 size={12}/>
            </motion.button>
          </div>
          <div className="photo-info">
            {img.caption_short&&<p className="photo-caption">{img.caption_short}</p>}
            <div className="photo-meta-row">
              {img.quality_level&&img.quality_level!=="Processing..."&&
                <span className="meta-chip" style={{color:q.c}}>◆ {img.quality_level}</span>}
              {img.aesthetic_score>0&&
                <span className="meta-chip" style={{color:"#fbbf24"}}>★ {Number(img.aesthetic_score).toFixed(1)}</span>}
              {img.dominant_emotion&&img.dominant_emotion!=="neutral"&&
                <span className="meta-chip">{emo} {img.dominant_emotion}</span>}
              {img.user_tags?.length>0&&
                <span className="meta-chip" style={{color:"#86efac"}}>#{img.user_tags[0]}{img.user_tags.length>1?` +${img.user_tags.length-1}`:""}</span>}
            </div>
          </div>
        </div>
      )}
    </motion.div>
  );
}

/* ══════════════════════════════════════════════════════════════════════════ */
/* LIGHTBOX                                                                    */
/* ══════════════════════════════════════════════════════════════════════════ */
function Lightbox({ list, idx, onClose, onShift, onFav, onDel, onAddTag, onRemoveTag, albums }) {
  const img=list[idx]; if(!img) return null;
  const q=qInfo(img.quality_level); const emo=EMO[img.dominant_emotion]||"😐";
  const [tagInput, setTagInput] = useState("");
  const [imgTags,  setImgTags]  = useState(img.user_tags||[]);
  const [addAlbum, setAddAlbum] = useState("");
  const [note,     setNote]     = useState(img.photo_note||"");
  const [noteEditing, setNoteEditing] = useState(false);
  const [noteSaving,  setNoteSaving]  = useState(false);

  // Always fetch latest tags + note from server when photo changes
  useEffect(()=>{
    setTagInput(""); setNoteEditing(false);
    setImgTags(img.user_tags||[]);
    setNote(img.photo_note||"");
    axios.get(`${API}/photo/${img.id}/tags`)
      .then(r => setImgTags(r.data.tags||[]))
      .catch(()=>{});
    axios.get(`${API}/photo/${img.id}/note`)
      .then(r => setNote(r.data.note||""))
      .catch(()=>{});
  }, [img.id]);

  const saveNote = async () => {
    setNoteSaving(true);
    const fd=new FormData(); fd.append("note", note);
    try { await axios.post(`${API}/photo/${img.id}/note`, fd); setNoteEditing(false); }
    catch(e) { alert("Failed to save note"); }
    setNoteSaving(false);
  };

  const doAddTag = async () => {
    if (!tagInput.trim()) return;
    const newTags = await onAddTag(img.id, tagInput.trim());
    if (newTags) setImgTags(newTags);
    else setImgTags(p=>[...new Set([...p, tagInput.trim().toLowerCase()])]);
    setTagInput("");
  };
  const doRemoveTag = async tag => {
    const newTags = await onRemoveTag(img.id, tag);
    if (newTags) setImgTags(newTags);
    else setImgTags(p=>p.filter(t=>t!==tag));
  };
  const doAddToAlbum = async () => {
    if (!addAlbum) return;
    const fd=new FormData(); fd.append("image_ids",img.id);
    try { await axios.post(`${API}/batch/album`,fd.set?fd:(()=>{fd.append("album_id",addAlbum);return fd;})()); } catch {}
    setAddAlbum("");
  };

  return (
    <motion.div className="lb-bg" initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}} onClick={onClose}>
      <motion.div className="lb-box" initial={{scale:.96}} animate={{scale:1}} exit={{scale:.96}} onClick={e=>e.stopPropagation()}>
        <div className="lb-left">
          {idx>0&&<button className="lb-nav lb-nav--l" onClick={()=>onShift(-1)}><ChevronLeft size={20}/></button>}
          <AnimatePresence mode="wait">
            <motion.img key={img.id??idx} src={imgUrl(img.filename)||BLANK} alt="" className="lb-img"
              initial={{opacity:0,x:16}} animate={{opacity:1,x:0}} exit={{opacity:0,x:-16}} transition={{duration:.18}}
              onError={e=>{e.target.onerror=null;e.target.src=BLANK;}}/>
          </AnimatePresence>
          {idx<list.length-1&&<button className="lb-nav lb-nav--r" onClick={()=>onShift(1)}><ChevronRight size={20}/></button>}
          <div className="lb-counter">{idx+1} / {list.length}</div>
          <div className="lb-hint">← → · ESC</div>
        </div>

        <div className="lb-right">
          <div className="lb-right-head">
            <h3 className="lb-right-title">Details</h3>
            <div className="lb-head-btns">
              <button className="lb-hbtn" onClick={()=>onFav(img.id)} style={{color:img.is_favorite?"#f43f5e":undefined}}>
                <Heart size={15} fill={img.is_favorite?"#f43f5e":"none"} stroke={img.is_favorite?"#f43f5e":"currentColor"}/>
              </button>
              <button className="lb-hbtn" onClick={e=>onDel(img.id,e)}><Trash2 size={15}/></button>
              <button className="lb-hbtn lb-close" onClick={onClose}><X size={17}/></button>
            </div>
          </div>

          <div className="lb-scroll">
            {img.caption_short&&(
              <DB icon={<Camera size={12}/>} label="AI Caption" c="#818cf8">
                <p className="db-caption">{img.caption_short}</p>
                {img.caption_detailed&&img.caption_detailed!==img.caption_short&&<p className="db-caption-sub">{img.caption_detailed}</p>}
              </DB>
            )}
            {img.quality_score>0&&(
              <DB icon={<TrendingUp size={12}/>} label="Quality" c={q.c}>
                <div className="q-wrap">
                  <div className="q-circle" style={{borderColor:q.c}}>
                    <span style={{color:q.c,fontSize:18,fontWeight:800}}>{Math.round(img.quality_score)}</span>
                  </div>
                  <div className="q-right">
                    <p className="q-level" style={{color:q.c}}>{img.quality_level}</p>
                    {img.sharpness>0&&<><QBar label="Sharpness" val={img.sharpness} c={q.c}/><QBar label="Exposure" val={img.exposure} c={q.c}/><QBar label="Contrast" val={img.contrast} c={q.c}/><QBar label="Composition" val={img.composition} c={q.c}/></>}
                  </div>
                </div>
              </DB>
            )}
            {img.aesthetic_score>0&&(
              <DB icon={<Star size={12}/>} label="Aesthetic" c="#fbbf24">
                <div className="aes-row"><span className="aes-score">★ {Number(img.aesthetic_score).toFixed(1)}</span>{img.aesthetic_rating&&<span className="aes-rating">{img.aesthetic_rating}</span>}</div>
              </DB>
            )}
            {img.dominant_emotion&&img.dominant_emotion!=="neutral"&&(
              <DB icon={<Smile size={12}/>} label="Emotion" c={EMO_COLORS[img.dominant_emotion]||"#f9a8d4"}>
                <div className="emo-row">
                  <span style={{fontSize:30}}>{EMO[img.dominant_emotion]}</span>
                  <div><p className="emo-name">{img.dominant_emotion}</p>{img.face_emotion_count>0&&<p className="emo-sub">{img.face_emotion_count} face{img.face_emotion_count>1?"s":""}</p>}</div>
                </div>
              </DB>
            )}
            {img.ocr_text_enhanced?.trim()&&(
              <DB icon={<Type size={12}/>} label="Text in Photo" c="#22d3ee">
                <p className="ocr-text">{img.ocr_text_enhanced}</p>
              </DB>
            )}

            {/* Tags block */}
            <DB icon={<FileText size={12}/>} label="Personal Note" c="#fcd34d">
              {noteEditing ? (
                <div>
                  <textarea className="form-textarea" value={note} onChange={e=>setNote(e.target.value)}
                    placeholder="Write a personal note about this photo…" rows={3} autoFocus
                    style={{marginBottom:6,fontSize:12}}/>
                  <div style={{display:"flex",gap:6}}>
                    <button className="btn-sm btn-sm--primary" onClick={saveNote} disabled={noteSaving}>
                      {noteSaving?"Saving…":"Save Note"}
                    </button>
                    <button className="btn-sm" onClick={()=>setNoteEditing(false)}>Cancel</button>
                  </div>
                </div>
              ) : (
                <div>
                  {note ? <p style={{fontSize:12,color:"var(--tx)",lineHeight:1.5,marginBottom:6}}>{note}</p>
                        : <p style={{fontSize:11,color:"var(--mu)"}}>No note yet</p>}
                  <button className="btn-sm" style={{fontSize:10,padding:"3px 8px"}} onClick={()=>setNoteEditing(true)}>
                    ✏ {note?"Edit Note":"Add Note"}
                  </button>
                </div>
              )}
            </DB>

            <DB icon={<Hash size={12}/>} label="Tags" c="#86efac">
              <div className="tag-chips">
                {imgTags.map(t=>(
                  <span key={t} className="tag-chip-item">
                    #{t}
                    <button className="tag-chip-x" onClick={()=>doRemoveTag(t)}><X size={8}/></button>
                  </span>
                ))}
                {imgTags.length===0&&<span style={{fontSize:10,color:"var(--mu)"}}>No tags yet</span>}
              </div>
              <div className="tag-add-row">
                <input className="tag-add-input" placeholder="Add tag…" value={tagInput} onChange={e=>setTagInput(e.target.value)} onKeyDown={e=>e.key==="Enter"&&doAddTag()}/>
                <button className="tag-add-btn" onClick={doAddTag}><Plus size={12}/></button>
              </div>
            </DB>

            {/* Add to album */}
            {albums?.length>0&&(
              <DB icon={<Layers size={12}/>} label="Add to Album" c="#a78bfa">
                <div className="tag-add-row">
                  <select className="batch-select" style={{flex:1}} value={addAlbum} onChange={e=>setAddAlbum(e.target.value)}>
                    <option value="">Choose album…</option>
                    {albums.map(a=><option key={a.id} value={a.id}>{a.title}</option>)}
                  </select>
                  <button className="tag-add-btn" onClick={doAddToAlbum}><Plus size={12}/></button>
                </div>
              </DB>
            )}

            {img.scene_label&&(
              <DB icon={<Tag size={12}/>} label="Objects" c="#86efac">
                <div className="tag-wrap">{img.scene_label.split(",").map((t,i)=><span key={i} className="obj-tag">{t.trim()}</span>)}</div>
              </DB>
            )}
            <DB icon={<Info size={12}/>} label="Metadata" c="#6b7280">
              <div className="meta-stack">
                {img.timestamp&&<MRow k="Captured" v={new Date(img.timestamp).toLocaleString()}/>}
                {img.width&&<MRow k="Dimensions" v={`${img.width} × ${img.height}`}/>}
                {img.person_count>0&&<MRow k="People" v={`${img.person_count} detected`}/>}
                {img.score>0&&<MRow k="Relevance" v={`${Math.round(img.score)}%`}/>}
              </div>
            </DB>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}

function QBar({label,val,c}){ return <div className="qbar-row"><span className="qbar-label">{label}</span><div className="qbar-track"><div className="qbar-fill" style={{width:`${Math.min(100,(val||0)*100)}%`,background:c}}/></div><span className="qbar-num">{Math.round((val||0)*100)}</span></div>; }
function MRow({k,v}){ return <div className="mrow"><span className="mrow-k">{k}</span><span className="mrow-v">{v}</span></div>; }
function DB({icon,label,c,children}){ return <div className="db" style={{"--dc":c}}><div className="db-head">{React.cloneElement(icon,{color:c})}<span className="db-label">{label}</span></div><div className="db-body">{children}</div></div>; }

/* Person / Album components */
function PersonCard({p,onClick}){ return <motion.div className="person-card" whileHover={{y:-3}} onClick={onClick}><div className="person-avatar">{p.cover?<img src={imgUrl(p.cover)} alt="" onError={e=>e.target.src=BLANK}/>:<Users size={26} strokeWidth={1} color="#444"/>}</div><p className="person-name">{p.name}</p><p className="person-count">{p.count||p.face_count||0} photos</p></motion.div>; }

function PersonModal({data,onClose,onRename,onOpen}){
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(data.name || "");
  const imgs = data.images || [];
  const isDefault = /^Person \d+$/.test(data.name || "");

  const doSave = () => {
    if (name.trim()) { onRename(data.id, name.trim()); setEditing(false); }
  };

  return (
    <ModalWrap onClose={onClose} wide>
      <div className="modal-head">
        <div>
          <h2 className="modal-title">{data.name}</h2>
          <p className="modal-sub">{data.face_count||data.count||0} photos</p>
        </div>
        <div style={{display:"flex",gap:6,alignItems:"center"}}>
          <button className="btn-sm btn-sm--primary" onClick={()=>{setEditing(true);setName(isDefault?"":data.name);}}>
            ✏ Rename
          </button>
          <button className="lb-hbtn lb-close" onClick={onClose}><X size={17}/></button>
        </div>
      </div>

      {isDefault && !editing && (
        <div style={{background:"rgba(250,204,21,.07)",border:"1px solid rgba(250,204,21,.2)",borderRadius:8,padding:"10px 14px",marginBottom:14}}>
          <p style={{fontSize:12,color:"rgba(250,204,21,.9)"}}>
            💡 <strong>Give this person a name</strong> so you can search for their photos. Click <strong>✏ Rename</strong> above.
          </p>
        </div>
      )}

      {editing && (
        <div style={{background:"var(--s2)",border:"1px solid var(--br)",borderRadius:10,padding:"14px 16px",marginBottom:14}}>
          <p style={{fontSize:11,color:"var(--mu)",marginBottom:8,fontFamily:"var(--mono)",letterSpacing:".07em"}}>
            ENTER PERSON'S NAME
          </p>
          <div style={{display:"flex",gap:8,alignItems:"center"}}>
            <input
              className="rename-in"
              style={{flex:1,fontSize:15,padding:"10px 14px"}}
              value={name}
              onChange={e=>setName(e.target.value)}
              onKeyDown={e=>{if(e.key==="Enter")doSave(); if(e.key==="Escape")setEditing(false);}}
              placeholder="e.g. Vijay, Shah Rukh Khan…"
              autoFocus
            />
            <button className="btn-sm btn-sm--primary" style={{padding:"10px 18px",fontSize:13}} onClick={doSave} disabled={!name.trim()}>
              Save
            </button>
            <button className="btn-sm" onClick={()=>setEditing(false)}>Cancel</button>
          </div>
          <p style={{fontSize:10,color:"#555",marginTop:8}}>
            After saving, you can search for this name in the search bar to find all their photos.
          </p>
        </div>
      )}

      {data.images===null
        ? <div className="loader-wrap" style={{minHeight:140}}><div className="loader-ring"/></div>
        : imgs.length>0
          ? <ImgGrid list={imgs} onOpen={i=>onOpen(imgs,i)} onFav={()=>{}} onDel={()=>{}}/>
          : <p className="empty-sub">No photos for this person</p>
      }
    </ModalWrap>
  );
}

function AlbumCard({a,onClick,onDelete}){
  return (
    <motion.div className="album-card" whileHover={{y:-3,scale:1.01}} onClick={onClick}>
      {a.cover
        ? <img src={imgUrl(a.cover)} alt="" className="album-cover" onError={e=>e.target.src=BLANK}/>
        : <div className="album-cover-ph"><FolderPlus size={32} strokeWidth={1} color="#555"/><span style={{fontSize:10,color:"#555",marginTop:4}}>Empty</span></div>
      }
      {a.thumbnails?.length>1&&<div className="album-strip">{a.thumbnails.slice(1,5).map((fn,i)=><img key={i} src={imgUrl(fn)} alt="" className="album-strip-img" onError={e=>e.target.style.display="none"}/>)}</div>}
      <div className="album-info">
        <h3 className="album-title">{a.title}</h3>
        <div className="album-meta">
          {a.type==="manual"
            ? <span className="tag-sm tag-sm--manual">✎ manual</span>
            : <span className="tag-sm tag-sm--accent">📅 event</span>
          }
          {/* Only show date badge if it adds info beyond the title (i.e. different month/year span) */}
          {a.type!=="manual" && a.date && !a.title.includes(a.date.split(" ")[0]) &&
            <span className="tag-sm">{a.date}</span>
          }
          <span className="tag-sm" style={{
            color: a.count===0 ? "rgba(239,68,68,.7)" : undefined,
            background: a.count===0 ? "rgba(239,68,68,.08)" : undefined,
          }}>
            {a.count===0 ? "⚠ empty" : `📸 ${a.count}`}
          </span>
        </div>
        {a.count===0 && a.type==="manual" && (
          <p style={{fontSize:9,color:"rgba(96,165,250,.6)",marginTop:4,fontFamily:"var(--mono)"}}>
            select photos → batch bar → add to album
          </p>
        )}
      </div>
      {onDelete&&<button className="album-del-btn" onClick={e=>{e.stopPropagation();onDelete();}} title="Delete album"><X size={12}/></button>}
    </motion.div>
  );
}

function AlbumModal({data,onClose,onOpen,onRename,onDelete}){
  const imgs=data.images||[];
  const [editing,setEditing]=React.useState(false);
  const [title,setTitle]=React.useState(data.title||"");
  const [desc,setDesc]=React.useState(data.description||"");
  const saveRename=()=>{ onRename&&onRename(data.id,title,desc); setEditing(false); };
  return (
    <ModalWrap onClose={onClose} wide>
      <div className="modal-head">
        <div>
          <h2 className="modal-title">{data.title}</h2>
          <p className="modal-sub">{data.image_count||data.count||0} photos{data.date?` · ${data.date}`:""}</p>
        </div>
        <div style={{display:"flex",gap:6,alignItems:"center"}}>
          <button className="btn-sm" onClick={()=>setEditing(e=>!e)}>✏ Rename</button>
          {data.type==="manual"&&onDelete&&(
            <button className="btn-sm" style={{color:"#f87171",borderColor:"rgba(248,113,113,.3)"}}
              onClick={()=>{ if(window.confirm(`Delete album "${data.title}"? Photos are kept.`)){onDelete(data.id);onClose();} }}>
              🗑 Delete
            </button>
          )}
          <button className="lb-hbtn lb-close" onClick={onClose}><X size={17}/></button>
        </div>
      </div>
      {editing&&(
        <div className="rename-row" style={{marginBottom:12}}>
          <input className="rename-in" value={title} onChange={e=>setTitle(e.target.value)}
            onKeyDown={e=>e.key==="Enter"&&saveRename()} placeholder="Album name…" autoFocus/>
          <input className="rename-in" value={desc} onChange={e=>setDesc(e.target.value)}
            placeholder="Description (optional)…" style={{marginLeft:6}}/>
          <button className="btn-sm btn-sm--primary" onClick={saveRename} style={{marginLeft:6}}>Save</button>
          <button className="btn-sm" onClick={()=>setEditing(false)} style={{marginLeft:4}}>Cancel</button>
        </div>
      )}
      {!editing&&data.description&&(
        <div className="album-desc"><span className="album-desc-lbl">DESCRIPTION</span><p>{data.description}</p></div>
      )}
      {data.images===null
        ? <div className="loader-wrap" style={{minHeight:140}}><div className="loader-ring"/></div>
        : imgs.length>0
          ? <ImgGrid list={imgs} onOpen={i=>onOpen(imgs,i)} onFav={()=>{}} onDel={()=>{}}/>
          : (
            <div style={{textAlign:"center",padding:"40px 20px",color:"var(--mu)"}}>
              <FolderPlus size={40} strokeWidth={1} style={{marginBottom:12,opacity:.4}}/>
              <p style={{fontSize:14,marginBottom:6}}>This album is empty</p>
              <p style={{fontSize:12,color:"#444"}}>
                Go to <strong>Timeline</strong> → select photos (checkbox icon) →
                choose this album in the batch bar → click the layers icon
              </p>
            </div>
          )
      }
    </ModalWrap>
  );
}

function ModalWrap({children,onClose,wide}){
  return <motion.div className="modal-bg" initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}} onClick={onClose}><motion.div className="modal-box" style={{maxWidth:wide?960:680}} initial={{scale:.95,y:12}} animate={{scale:1,y:0}} exit={{scale:.95}} onClick={e=>e.stopPropagation()}>{children}</motion.div></motion.div>;
}

function CoOccurrencePanel({faces, onSearch}) {
  const [selected, setSelected] = React.useState([]);
  const toggle = id => setSelected(p => p.includes(id) ? p.filter(x=>x!==id) : [...p, id]);
  return (
    <div className="create-album-form">
      <p style={{fontSize:12,color:"var(--mu)",marginBottom:10}}>Select 2+ people to find photos where they all appear together:</p>
      <div style={{display:"flex",flexWrap:"wrap",gap:8,marginBottom:14,maxHeight:240,overflowY:"auto"}}>
        {(faces||[]).map(p=>(
          <button key={p.id}
            className={"person-chip" + (selected.includes(p.id)?" person-chip--on":"")}
            onClick={()=>toggle(p.id)}>
            {p.cover && <img src={imgUrl(p.cover)} alt="" style={{width:24,height:24,borderRadius:"50%",objectFit:"cover",marginRight:6}}/>}
            {p.name||`Person ${p.id}`}
          </button>
        ))}
        {(!faces||faces.length===0) && <p style={{fontSize:12,color:"var(--mu)"}}>No named people yet. Go to People section first.</p>}
      </div>
      <button className="btn-sm btn-sm--primary" disabled={selected.length<2} onClick={()=>onSearch(selected)}>
        <GitMerge size={12}/> Find Co-appearing Photos ({selected.length} selected)
      </button>
    </div>
  );
}

function EmotionTimelineChart() {
  const [data, setData] = React.useState(null);
  React.useEffect(()=>{
    axios.get(`${API}/emotion-timeline`).then(r=>setData(r.data)).catch(()=>{});
  },[]);

  if (!data) return <p style={{fontSize:12,color:"var(--mu)",padding:8}}>Loading…</p>;
  if (!data.months?.length) return (
    <div style={{padding:"12px 0"}}>
      <p style={{fontSize:12,color:"var(--mu)"}}>No emotion timeline data yet.</p>
      <p style={{fontSize:11,color:"#555",marginTop:4}}>
        Emotions are detected in the background after uploading. Install <code style={{background:"var(--s3)",padding:"1px 5px",borderRadius:3}}>pip install fer</code> for best results, then click <strong>Re-index AI</strong>.
      </p>
    </div>
  );

  // Show server hint if only neutral detected
  const serverHint = data.hint;

  const EMO_COL = {happy:"#4ade80",sad:"#60a5fa",angry:"#f87171",neutral:"#a8a29e",surprised:"#fbbf24",disgusted:"#a78bfa",fearful:"#f9a8d4"};
  const EMO_EMOJI = {happy:"😊",sad:"😢",angry:"😠",neutral:"😐",surprised:"😲",disgusted:"🤢",fearful:"😨"};

  const months = data.months;
  const n = months.length;
  // Filter to emotions that actually have data (exclude neutral-only if others exist)
  const showEmos = data.emotions.filter(e => (data.series[e]||[]).some(v=>v>0));
  const hasRealData = showEmos.some(e => e !== "neutral");

  // Chart dimensions — extra bottom padding for legend
  const W=560, CHART_H=160, LEG_H=28, PAD={l:36,r:16,t:12,b:32};
  const chartW = W - PAD.l - PAD.r;
  const chartH = CHART_H - PAD.t - PAD.b;
  const maxVal = Math.max(1, ...showEmos.flatMap(e => data.series[e]||[]));

  const xScale = i => PAD.l + (n <= 1 ? chartW/2 : (i / (n-1)) * chartW);
  const yScale = v => PAD.t + chartH - (v / maxVal) * chartH;
  const makePath = series => series.map((v,i) =>
    `${i===0?"M":"L"}${xScale(i).toFixed(1)},${yScale(v).toFixed(1)}`
  ).join(" ");

  // X labels: show at most 8, evenly spaced
  const maxLabels = Math.min(8, n);
  const labelIndices = n <= maxLabels
    ? months.map((_,i)=>i)
    : Array.from({length:maxLabels}, (_,k) => Math.round(k*(n-1)/(maxLabels-1)));

  return (
    <div style={{overflowX:"auto"}}>
      {(serverHint || !hasRealData) && (
        <p style={{fontSize:11,color:"#f59e0b",marginBottom:8,background:"rgba(245,158,11,.08)",padding:"6px 10px",borderRadius:6,border:"1px solid rgba(245,158,11,.2)"}}>
          ⚠️ {serverHint || 'Only "neutral" detected — run pip install fer then Re-index AI.'}
        </p>
      )}
      <svg viewBox={`0 0 ${W} ${CHART_H + LEG_H}`} style={{width:"100%",maxWidth:W,display:"block"}}>
        {/* Y grid + labels */}
        {[0,0.25,0.5,0.75,1].map((f,i)=>{
          const y = PAD.t + chartH*(1-f);
          return (
            <g key={i}>
              <line x1={PAD.l} x2={W-PAD.r} y1={y} y2={y} stroke="#ffffff08" strokeWidth={1}/>
              <text x={PAD.l-4} y={y+3} fontSize={7} fill="#555" textAnchor="end">{Math.round(maxVal*f)}</text>
            </g>
          );
        })}
        {/* Axes */}
        <line x1={PAD.l} x2={PAD.l} y1={PAD.t} y2={PAD.t+chartH} stroke="#333" strokeWidth={1}/>
        <line x1={PAD.l} x2={W-PAD.r} y1={PAD.t+chartH} y2={PAD.t+chartH} stroke="#333" strokeWidth={1}/>
        {/* Emotion lines + dots */}
        {showEmos.map(e=>(
          <g key={e}>
            <path d={makePath(data.series[e]||[])} fill="none" stroke={EMO_COL[e]||"#888"} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round"/>
            {(data.series[e]||[]).map((v,i)=> v>0 && (
              <circle key={i} cx={xScale(i)} cy={yScale(v)} r={3} fill={EMO_COL[e]||"#888"} opacity={0.85}/>
            ))}
          </g>
        ))}
        {/* X axis labels */}
        {labelIndices.map(i=>(
          <text key={i} x={xScale(i)} y={PAD.t+chartH+14} fontSize={7} fill="#555" textAnchor="middle">
            {months[i]}
          </text>
        ))}
        {/* Legend — below chart, wrapping row */}
        {showEmos.map((e,i)=>{
          const cols = Math.min(showEmos.length, 4);
          const lx = PAD.l + (i % cols) * 130;
          const ly = CHART_H + Math.floor(i/cols)*14 + 4;
          return (
            <g key={e} transform={`translate(${lx},${ly})`}>
              <rect width={8} height={8} fill={EMO_COL[e]||"#888"} rx={2}/>
              <text x={11} y={7} fontSize={9} fill="#aaa">{EMO_EMOJI[e]} {e}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function PeopleFreqChart({people}) {
  if(!people?.length) return <p style={{fontSize:12,color:"var(--mu)",padding:8}}>No people data yet.</p>;
  const top = people.slice(0,12);
  const maxN = Math.max(...top.map(p=>p.count),1);
  return (
    <div style={{display:"flex",flexDirection:"column",gap:6}}>
      {top.map((p,i)=>(
        <div key={p.id} style={{display:"flex",alignItems:"center",gap:8}}>
          <span style={{fontSize:11,color:"var(--mu)",width:20,textAlign:"right"}}>{i+1}</span>
          <span style={{fontSize:12,color:"var(--tx)",width:120,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{p.name||`Person ${p.id}`}</span>
          <div style={{flex:1,height:14,background:"var(--s3)",borderRadius:4,overflow:"hidden"}}>
            <div style={{width:`${(p.count/maxN)*100}%`,height:"100%",background:"linear-gradient(90deg,#818cf8,#a78bfa)",borderRadius:4,transition:"width .4s"}}/>
          </div>
          <span style={{fontSize:11,color:"#818cf8",width:30,textAlign:"right"}}>{p.count}</span>
        </div>
      ))}
    </div>
  );
}

function StatsPage({stats, peopleFreq}){
  if(!stats) return <Empty icon={BarChart3} msg="Loading stats…"/>;
  const cards=[{l:"Photos",v:stats.total_images,c:"#818cf8"},{l:"Faces",v:stats.total_faces,c:"#a78bfa"},{l:"People",v:stats.total_people,c:"#f472b6"},{l:"Albums",v:stats.total_albums,c:"#fbbf24"},{l:"Favorites",v:stats.total_favorites,c:"#4ade80"},{l:"Indexed",v:stats.indexed_vectors,c:"#22d3ee"}];
  return (
    <div className="stats-wrap">
      <PageHead title="Statistics"/>
      <div className="stats-cards">{cards.map((c,i)=><motion.div key={i} className="stat-card" whileHover={{y:-3}} style={{"--sc":c.c}}><p className="stat-lbl">{c.l}</p><p className="stat-val" style={{color:c.c}}>{c.v??0}</p></motion.div>)}</div>

      <div className="stats-sec">
        <p className="stats-sec-lbl">EMOTION TIMELINE</p>
        <EmotionTimelineChart/>
      </div>

      {peopleFreq?.length>0&&(
        <div className="stats-sec">
          <p className="stats-sec-lbl">MOST PHOTOGRAPHED PEOPLE</p>
          <PeopleFreqChart people={peopleFreq}/>
        </div>
      )}

      {stats.color_distribution?.length>0&&<div className="stats-sec"><p className="stats-sec-lbl">COLOR DISTRIBUTION</p><div className="clr-dist">{stats.color_distribution.map((d,i)=><div key={i} className="clr-dist-item"><div className="clr-dist-dot" style={{background:d.color==="gray"?"#888":d.color==="white"?"#ddd":d.color}}/><span>{d.color}</span><span className="clr-dist-n">×{d.count}</span></div>)}</div></div>}
      {stats.top_user_tags?.length>0&&<div className="stats-sec"><p className="stats-sec-lbl">YOUR TAGS</p><div className="tag-cloud">{stats.top_user_tags.map((t,i)=><span key={i} className="tag-cloud-item" style={{borderColor:"#86efac44",color:"#86efac"}}>#{t.tag}<span className="tag-n" style={{color:"#86efac99"}}> ×{t.count}</span></span>)}</div></div>}
      {stats.top_tags?.length>0&&<div className="stats-sec"><p className="stats-sec-lbl">AI DETECTED OBJECTS</p><div className="tag-cloud">{stats.top_tags.map((t,i)=><span key={i} className="tag-cloud-item">{t.tag}<span className="tag-n"> ×{t.count}</span></span>)}</div></div>}
    </div>
  );
}

function ToolsAccordion({reindexing, onEmotions, onColors, onNames, onCaptions, onRecaptionAll, onCleanup}) {
  const [open, setOpen] = React.useState(false);
  return (
    <div className="tools-accordion">
      <button className="tools-accordion-btn" onClick={()=>setOpen(p=>!p)}>
        <span style={{display:"flex",alignItems:"center",gap:6}}>
          <SlidersHorizontal size={11}/> AI Tools
        </span>
        <motion.span animate={{rotate:open?180:0}} transition={{duration:.2}}>
          <ChevronDown size={11}/>
        </motion.span>
      </button>
      <AnimatePresence>
        {open && (
          <motion.div initial={{height:0,opacity:0}} animate={{height:"auto",opacity:1}} exit={{height:0,opacity:0}}
            style={{overflow:"hidden"}}>
            <div style={{display:"flex",flexDirection:"column",gap:4,paddingTop:4}}>
              {[
                {label:"Fix Captions",    color:"rgba(167,139,250,.9)", bg:"rgba(167,139,250,.07)", border:"rgba(167,139,250,.2)", fn:onCaptions,     title:"Caption images that are missing descriptions"},
                {label:"Re-caption All",  color:"rgba(129,140,248,.85)",bg:"rgba(129,140,248,.07)", border:"rgba(129,140,248,.2)", fn:onRecaptionAll, title:"Re-run BLIP on every image (slow — use after model upgrade)"},
                {label:"Fix Emotions",    color:"rgba(250,204,21,.8)",  bg:"rgba(250,204,21,.07)",  border:"rgba(250,204,21,.2)",  fn:onEmotions,     title:"Re-detect emotions on all photos"},
                {label:"Fix Colors",      color:"rgba(52,211,153,.85)", bg:"rgba(52,211,153,.07)",  border:"rgba(52,211,153,.2)",  fn:onColors,       title:"Recompute dominant colors for color search"},
                {label:"Auto-Name",       color:"rgba(96,165,250,.85)", bg:"rgba(96,165,250,.07)",  border:"rgba(96,165,250,.2)",  fn:onNames,        title:"Auto-detect names from photo captions"},
                {label:"Clean Albums",    color:"rgba(248,113,113,.8)", bg:"rgba(248,113,113,.07)", border:"rgba(248,113,113,.2)", fn:onCleanup,      title:"Delete empty auto-generated albums"},
              ].map(({label,color,bg,border,fn,title})=>(
                <button key={label} disabled={reindexing}
                  className="reindex-btn"
                  title={title}
                  style={{background:bg,borderColor:border,color,fontSize:10,padding:"7px"}}
                  onClick={fn}>
                  {label}
                </button>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function PageHead({title,count,unit="photos",extra}){ return <div className="page-head"><h1 className="page-title">{title}</h1>{count>0&&<span className="page-count">{count} {unit}</span>}{extra&&<div style={{marginLeft:"auto"}}>{extra}</div>}</div>; }
function Empty({icon:Icon,msg,sub,onClick}){ return <div className={`empty ${onClick?"empty--click":""}`} onClick={onClick}><Icon size={44} strokeWidth={1} color="#333"/><p className="empty-msg">{msg}</p>{sub&&<p className="empty-sub">{sub}</p>}{onClick&&<p className="empty-cta">CLICK TO UPLOAD</p>}</div>; }
function Fade({children}){ return <motion.div initial={{opacity:0,y:6}} animate={{opacity:1,y:0}} exit={{opacity:0}} transition={{duration:.18}}>{children}</motion.div>; }

/* ══════════════════════════════════════════════════════════════════════════ */
/* CSS                                                                         */
/* ══════════════════════════════════════════════════════════════════════════ */
function CSS() {
  return <style>{`
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300..900;1,14..32,300..900&family=JetBrains+Mono:wght@400;500&display=swap');

*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

:root{
  /* Background layers */
  --bg0:#000000;
  --bg1:#0a0a0a;
  --bg2:#111111;
  --bg3:#161616;
  --bg4:#1c1c1c;

  /* Borders */
  --b1:rgba(255,255,255,.06);
  --b2:rgba(255,255,255,.1);
  --b3:rgba(255,255,255,.15);

  /* Accent */
  --ac:#6366f1;
  --ac-l:#818cf8;
  --ac-d:#4f46e5;
  --ac-glow:rgba(99,102,241,.25);
  --ac-bg:rgba(99,102,241,.08);

  /* Text */
  --t1:#ffffff;
  --t2:#a1a1aa;
  --t3:#52525b;

  /* Status */
  --green:#22c55e;
  --red:#ef4444;
  --gold:#f59e0b;
  --sky:#38bdf8;
  --rose:#f43f5e;

  --font:'Inter',system-ui,sans-serif;
  --mono:'JetBrains Mono',monospace;
  --r:8px;
  --r2:12px;
  --r3:16px;
}

html,body,#root{height:100%;background:var(--bg0);color:var(--t1);font-family:var(--font);overflow:hidden;-webkit-font-smoothing:antialiased}

::-webkit-scrollbar{width:3px;height:3px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--b2);border-radius:3px}

/* ═══ LAYOUT ══════════════════════════════════════════════════════════════ */
.shell{display:flex;height:100vh;overflow:hidden;background:var(--bg0)}

/* ═══ SIDEBAR ═════════════════════════════════════════════════════════════ */
.sidebar{
  width:220px;flex-shrink:0;
  background:var(--bg1);
  border-right:1px solid var(--b1);
  display:flex;flex-direction:column;
  padding:0;overflow-y:auto;overflow-x:hidden;
}

/* Brand */
.brand{
  display:flex;align-items:center;gap:10px;
  padding:20px 16px 16px;
  border-bottom:1px solid var(--b1);
  flex-shrink:0;
}
.brand-mark{
  width:32px;height:32px;border-radius:8px;
  background:var(--ac);
  display:flex;align-items:center;justify-content:center;
  flex-shrink:0;color:#fff;
  box-shadow:0 0 0 1px rgba(99,102,241,.4),0 4px 12px rgba(99,102,241,.3);
}
.brand-name{font-size:14px;font-weight:700;letter-spacing:-.01em;color:var(--t1)}
.brand-sub{font-family:var(--mono);font-size:9px;color:var(--t3);letter-spacing:.06em;margin-top:1px}

/* Nav */
.nav-list{display:flex;flex-direction:column;padding:8px;gap:1px;flex-shrink:0}
.nav-btn{
  display:flex;align-items:center;gap:9px;
  width:100%;padding:8px 10px;
  border-radius:var(--r);background:transparent;border:none;
  font-family:var(--font);font-size:13px;font-weight:500;
  color:var(--t3);cursor:pointer;text-align:left;
  transition:color .12s,background .12s;position:relative;
}
.nav-btn:hover{background:var(--bg3);color:var(--t2)}
.nav-btn--on{background:var(--bg3);color:var(--t1);font-weight:600}
.nav-btn--on .nav-ic{color:var(--ac-l)}
.nav-btn--trash:hover{color:var(--red);background:rgba(239,68,68,.06)}
.nav-ic{display:flex;flex-shrink:0;color:inherit}
.nav-pip{
  position:absolute;right:8px;
  width:4px;height:4px;border-radius:50%;background:var(--ac);
  box-shadow:0 0 6px var(--ac);
}
.badge{
  margin-left:auto;background:var(--red);color:#fff;
  font-size:10px;font-family:var(--mono);
  padding:1px 5px;border-radius:20px;line-height:1.5;
}

.sidebar-sep{height:1px;background:var(--b1);margin:6px 16px;flex-shrink:0}
.sidebar-section-label{
  display:flex;align-items:center;gap:5px;
  font-size:10px;font-weight:600;color:var(--t3);
  letter-spacing:.06em;text-transform:uppercase;
  padding:8px 16px 4px;flex-shrink:0;
}

/* Tag nav */
.tag-nav-list{display:flex;flex-direction:column;padding:0 8px;gap:1px}
.tag-nav-btn{
  display:flex;align-items:center;justify-content:space-between;
  width:100%;padding:6px 10px;border-radius:var(--r);
  background:transparent;border:none;
  font-family:var(--mono);font-size:11px;color:var(--t3);
  cursor:pointer;transition:all .12s;
}
.tag-nav-btn:hover{background:var(--bg3);color:var(--green)}
.tag-nav-btn--on{background:rgba(34,197,94,.07);color:var(--green)}
.tag-nav-count{font-size:9px;color:var(--t3)}

/* Sidebar footer */
.sidebar-foot{
  display:flex;flex-direction:column;gap:4px;
  margin-top:auto;padding:12px 8px;border-top:1px solid var(--b1);
  flex-shrink:0;
}
.reindex-btn{
  display:flex;align-items:center;justify-content:center;gap:6px;
  background:transparent;border:1px solid var(--b1);
  border-radius:var(--r);padding:8px;
  font-family:var(--font);font-size:12px;font-weight:500;
  color:var(--t3);cursor:pointer;transition:all .12s;
}
.reindex-btn:hover{border-color:var(--b2);color:var(--t2);background:var(--bg3)}
.reindex-btn:disabled{opacity:.35;cursor:default}
.reindex-btn--primary{
  background:var(--ac-bg);border-color:rgba(99,102,241,.3);
  color:var(--ac-l);font-weight:600;
}
.reindex-btn--primary:hover{background:rgba(99,102,241,.15);border-color:var(--ac)}

.reindex-msg{font-family:var(--mono);font-size:9px;color:var(--t3);text-align:center;padding:2px 4px;line-height:1.5}

.offline-pill{
  display:flex;align-items:center;gap:6px;
  background:rgba(34,197,94,.05);border:1px solid rgba(34,197,94,.12);
  border-radius:var(--r);padding:7px 10px;
  font-size:11px;font-weight:500;color:rgba(34,197,94,.7);
}
.dot-green{width:6px;height:6px;border-radius:50%;background:var(--green);box-shadow:0 0 6px var(--green);flex-shrink:0}

/* Tools accordion */
.tools-accordion{border:1px solid var(--b1);border-radius:var(--r);overflow:hidden;background:var(--bg2)}
.tools-accordion-btn{
  display:flex;align-items:center;justify-content:space-between;
  width:100%;padding:8px 10px;background:transparent;border:none;
  font-family:var(--font);font-size:11px;font-weight:500;color:var(--t3);cursor:pointer;
  transition:color .12s;
}
.tools-accordion-btn:hover{color:var(--t2)}

/* ═══ MAIN ════════════════════════════════════════════════════════════════ */
.main{flex:1;display:flex;flex-direction:column;min-width:0;overflow:hidden}

/* ═══ TOPBAR ══════════════════════════════════════════════════════════════ */
.topbar{
  background:var(--bg1);
  border-bottom:1px solid var(--b1);
  padding:10px 20px;
  display:flex;flex-direction:column;gap:8px;flex-shrink:0;
}

/* Mode pills */
.mode-strip{display:flex;gap:2px;align-items:center;flex-wrap:wrap}
.mode-btn{
  display:flex;align-items:center;gap:5px;
  padding:5px 10px;border-radius:6px;border:none;cursor:pointer;
  font-family:var(--font);font-size:11px;font-weight:500;
  background:transparent;color:var(--t3);transition:all .12s;
}
.mode-btn:hover{background:var(--bg3);color:var(--t2)}
.mode-btn--on{background:var(--ac);color:#fff;font-weight:600}
.voice-chip{
  margin-left:6px;display:flex;align-items:center;gap:4px;
  font-family:var(--mono);font-size:9px;letter-spacing:.05em;
  padding:4px 10px;border-radius:6px;
}
.voice-chip--rec{background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.2);color:var(--red)}
.voice-chip--processing{background:var(--ac-bg);border:1px solid rgba(99,102,241,.2);color:var(--ac-l)}
.voice-chip--error{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.15);color:#f87171}

/* Search bar */
.search-bar{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.search-wrap{
  flex:1;min-width:200px;
  display:flex;align-items:center;gap:8px;
  background:var(--bg2);border:1px solid var(--b1);
  border-radius:var(--r2);padding:0 12px;
  transition:border-color .15s;
}
.search-wrap:focus-within{border-color:rgba(99,102,241,.4)}
.search-ic{color:var(--t3);flex-shrink:0}
.search-in{
  flex:1;background:transparent;border:none;
  padding:9px 0;font-family:var(--font);font-size:13px;
  color:var(--t1);outline:none;
}
.search-in::placeholder{color:var(--t3)}
.mic-btn{background:transparent;border:none;cursor:pointer;color:var(--t3);padding:3px;display:flex;border-radius:5px;transition:color .12s}
.mic-btn:hover{color:var(--t2)}
.mic-btn--rec{color:var(--red)}
.mic-btn--proc{color:var(--ac-l)}

.img-search-label{
  display:flex;align-items:center;gap:7px;cursor:pointer;
  background:var(--bg2);border:1px solid var(--b1);
  border-radius:var(--r2);padding:8px 12px;
  font-size:12px;color:var(--t3);flex-shrink:0;transition:all .12s;
}
.img-search-label:hover{border-color:var(--b2);color:var(--t2)}
.img-search-thumb{width:20px;height:20px;border-radius:4px;object-fit:cover}

.color-strip{display:flex;gap:5px;align-items:center;flex-wrap:wrap}
.clr-dot{
  width:22px;height:22px;border-radius:50%;border:none;cursor:pointer;
  outline:2px solid transparent;outline-offset:2px;transition:all .12s;flex-shrink:0;
}
.clr-dot:hover{transform:scale(1.15)}
.clr-dot--on{outline-color:rgba(255,255,255,.5);transform:scale(1.2)}

.btn-go{
  background:var(--ac);color:#fff;border:none;border-radius:var(--r2);
  padding:8px 16px;font-family:var(--font);font-size:12px;font-weight:600;
  cursor:pointer;transition:background .12s;flex-shrink:0;
}
.btn-go:hover{background:var(--ac-d)}

.btn-batch{
  display:flex;align-items:center;gap:5px;
  background:var(--bg2);border:1px solid var(--b1);
  color:var(--t3);border-radius:var(--r2);
  padding:8px 12px;font-family:var(--font);font-size:12px;font-weight:500;
  cursor:pointer;transition:all .12s;flex-shrink:0;
}
.btn-batch:hover{border-color:var(--b2);color:var(--t2)}
.btn-batch--on{background:var(--ac-bg);border-color:rgba(99,102,241,.3);color:var(--ac-l)}

.btn-upload{
  display:flex;align-items:center;gap:5px;
  background:var(--bg2);border:1px solid var(--b2);
  color:var(--t2);border-radius:var(--r2);
  padding:8px 14px;font-family:var(--font);font-size:12px;font-weight:600;
  cursor:pointer;transition:all .12s;flex-shrink:0;
}
.btn-upload:hover{border-color:var(--b3);color:var(--t1);background:var(--bg3)}

/* Emotion filter bar */
.emo-filter-bar{display:flex;align-items:center;gap:5px;flex-wrap:wrap}
.emo-filter-label{font-family:var(--mono);font-size:9px;color:var(--t3);letter-spacing:.06em;flex-shrink:0}
.emo-filter-btn{
  display:flex;align-items:center;gap:4px;
  padding:4px 10px;border-radius:20px;
  border:1px solid var(--b1);background:transparent;
  color:var(--t3);font-family:var(--font);font-size:11px;font-weight:500;
  cursor:pointer;transition:all .12s;flex-shrink:0;
}
.emo-filter-btn:hover{border-color:var(--b2);color:var(--t2)}
.emo-filter-count{font-family:var(--mono);font-size:8px;opacity:.5;margin-left:2px}
.emo-filter-clear{
  display:flex;align-items:center;gap:4px;padding:4px 9px;
  border-radius:20px;border:1px solid rgba(239,68,68,.2);
  background:rgba(239,68,68,.06);color:var(--red);
  font-family:var(--font);font-size:11px;font-weight:500;cursor:pointer;transition:all .12s;
}

/* Emoji bar */
.emoji-quick-bar{display:flex;align-items:center;gap:3px;flex-wrap:wrap}
.emoji-quick-label{font-family:var(--mono);font-size:9px;color:var(--t3);letter-spacing:.06em;flex-shrink:0;margin-right:2px}
.emoji-quick-btn{
  background:var(--bg3);border:1px solid var(--b1);
  border-radius:6px;padding:3px 6px;font-size:14px;
  cursor:pointer;transition:all .12s;line-height:1;
}
.emoji-quick-btn:hover{background:var(--ac-bg);border-color:rgba(99,102,241,.3);transform:scale(1.12)}

/* ═══ BODY ════════════════════════════════════════════════════════════════ */
.body{flex:1;overflow-y:auto;padding:20px 20px 80px}
.page-head{display:flex;align-items:baseline;gap:10px;margin-bottom:18px;flex-wrap:wrap}
.page-title{font-size:18px;font-weight:700;letter-spacing:-.025em;color:var(--t1)}
.page-count{font-family:var(--mono);font-size:9px;color:var(--t3);letter-spacing:.05em}
.sel-count{font-family:var(--mono);font-size:9px;color:var(--ac-l);background:var(--ac-bg);padding:2px 8px;border-radius:20px;border:1px solid rgba(99,102,241,.2)}

/* On This Day */
.otd-banner{
  background:rgba(245,158,11,.04);
  border:1px solid rgba(245,158,11,.12);
  border-radius:var(--r3);margin-bottom:18px;overflow:hidden;
}
.otd-header{display:flex;align-items:center;justify-content:space-between;padding:12px 16px;cursor:pointer}
.otd-left{display:flex;align-items:center;gap:10px}
.otd-title{font-size:13px;font-weight:600;display:block}
.otd-sub{font-family:var(--mono);font-size:9px;color:rgba(245,158,11,.5);display:block;margin-top:1px}
.otd-toggle{background:none;border:none;cursor:pointer;color:var(--t3);display:flex;padding:3px;border-radius:5px}
.otd-year{padding:0 16px 14px}
.otd-year-label{font-family:var(--mono);font-size:9px;color:rgba(245,158,11,.5);letter-spacing:.06em;margin-bottom:8px;display:block}
.otd-strip{display:flex;gap:8px;overflow-x:auto;padding-bottom:4px}
.otd-thumb{flex-shrink:0;width:96px;cursor:pointer;border-radius:var(--r);overflow:hidden;border:1px solid var(--b1);transition:border-color .15s}
.otd-thumb:hover{border-color:rgba(245,158,11,.3)}
.otd-thumb img{width:100%;height:64px;object-fit:cover;display:block}
.otd-thumb-caption{font-size:9px;color:var(--t3);padding:4px 6px;line-height:1.4;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}

/* ═══ IMAGE GRID ══════════════════════════════════════════════════════════ */
.img-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(176px,1fr));gap:8px}
.img-grid--sm{grid-template-columns:repeat(auto-fill,minmax(124px,1fr));gap:6px}

/* Photo card */
.photo-card{
  position:relative;aspect-ratio:1;
  border-radius:var(--r2);overflow:hidden;cursor:pointer;
  background:var(--bg2);border:1px solid var(--b1);
  transition:border-color .15s;
}
.photo-card:hover{border-color:var(--b2)}
.photo-card--sel{border-color:var(--ac)!important;box-shadow:0 0 0 2px rgba(99,102,241,.3)!important}
.photo-img{width:100%;height:100%;object-fit:cover;display:block;transition:transform .25s}
.photo-card:hover .photo-img{transform:scale(1.04)}
.chip{position:absolute;font-family:var(--mono);font-size:8px;border-radius:4px;border:1px solid;padding:1px 5px;pointer-events:none;backdrop-filter:blur(4px)}
.chip--q{top:7px;left:7px}
.chip--emo{top:7px;right:7px;font-size:13px;border:none;background:none;padding:0}
.fav-dot{position:absolute;bottom:7px;right:7px;color:var(--rose);font-size:13px;pointer-events:none}
.batch-check{position:absolute;top:7px;left:7px}
.photo-hover{
  position:absolute;inset:0;opacity:0;transition:opacity .18s;
  background:linear-gradient(to top,rgba(0,0,0,.9) 0%,transparent 55%);
  display:flex;flex-direction:column;justify-content:flex-end;
}
.photo-card:hover .photo-hover{opacity:1}
.photo-actions{position:absolute;top:7px;right:7px;display:flex;gap:4px}
.ph-btn{
  width:28px;height:28px;border-radius:6px;border:none;cursor:pointer;
  display:flex;align-items:center;justify-content:center;transition:all .12s;
  backdrop-filter:blur(6px);
}
.ph-btn--fav{background:rgba(244,63,94,.7);color:#fff}
.ph-btn--del{background:rgba(0,0,0,.6);border:1px solid rgba(255,255,255,.1);color:#888}
.ph-btn--del:hover{background:rgba(239,68,68,.8);color:#fff;border-color:transparent}
.photo-info{padding:8px 8px 6px}
.photo-caption{font-size:10px;line-height:1.45;color:rgba(255,255,255,.9);margin-bottom:4px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.photo-meta-row{display:flex;flex-wrap:wrap;gap:3px}
.meta-chip{font-family:var(--mono);font-size:7.5px;color:rgba(255,255,255,.45);background:rgba(0,0,0,.4);padding:1px 5px;border-radius:3px;text-transform:capitalize}

/* ═══ BATCH BAR ═══════════════════════════════════════════════════════════ */
.batch-bar{
  position:absolute;bottom:0;left:0;right:0;
  background:rgba(10,10,10,.97);backdrop-filter:blur(12px);
  border-top:1px solid var(--b1);
  padding:10px 20px;
  display:flex;align-items:center;gap:10px;flex-wrap:wrap;z-index:50;
}
.batch-bar-left{display:flex;align-items:center;gap:7px}
.batch-count{display:flex;align-items:center;gap:5px;font-size:12px;font-weight:600;color:var(--ac-l)}
.batch-sm{background:var(--bg3);border:1px solid var(--b1);color:var(--t3);border-radius:6px;padding:4px 10px;font-family:var(--font);font-size:11px;font-weight:500;cursor:pointer;transition:all .12s}
.batch-sm:hover{border-color:var(--b2);color:var(--t2)}
.batch-bar-actions{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-left:auto}
.batch-action-btn{
  display:flex;align-items:center;gap:5px;padding:6px 12px;
  border-radius:var(--r);border:1px solid var(--b1);
  background:var(--bg2);color:var(--t2);
  font-family:var(--font);font-size:11px;font-weight:500;cursor:pointer;transition:all .12s;
}
.batch-action-btn:hover{border-color:var(--b2);color:var(--t1);background:var(--bg3)}
.batch-action-btn--fav:hover{border-color:rgba(244,63,94,.3);color:var(--rose)}
.batch-action-btn--del:hover{border-color:rgba(239,68,68,.3);color:var(--red)}
.batch-action-btn--album{border-color:rgba(56,189,248,.2);color:var(--sky)}
.batch-action-btn--album:hover{background:rgba(56,189,248,.08);border-color:rgba(56,189,248,.3)}
.batch-input-group{display:flex;gap:3px;align-items:center}
.batch-input{background:var(--bg2);border:1px solid var(--b1);border-radius:var(--r);padding:6px 10px;font-family:var(--font);font-size:11px;color:var(--t1);outline:none;width:108px;transition:border-color .12s}
.batch-input:focus{border-color:rgba(99,102,241,.4)}
.batch-select{background:var(--bg2);border:1px solid var(--b1);border-radius:var(--r);padding:6px 10px;font-family:var(--font);font-size:11px;color:var(--t2);outline:none;cursor:pointer}
.batch-select:focus{border-color:rgba(99,102,241,.4)}

/* ═══ LIGHTBOX ════════════════════════════════════════════════════════════ */
.lb-bg{
  position:fixed;inset:0;z-index:300;
  background:rgba(0,0,0,.9);backdrop-filter:blur(16px);
  display:flex;align-items:center;justify-content:center;padding:14px;
}
.lb-box{
  width:100%;max-width:1100px;max-height:92vh;
  background:var(--bg1);border:1px solid var(--b1);
  border-radius:16px;display:grid;grid-template-columns:1fr 300px;
  overflow:hidden;box-shadow:0 24px 80px rgba(0,0,0,.8);
}
.lb-left{position:relative;background:var(--bg0);display:flex;align-items:center;justify-content:center;min-height:380px;overflow:hidden}
.lb-img{max-width:100%;max-height:90vh;object-fit:contain;display:block}
.lb-nav{
  position:absolute;top:50%;transform:translateY(-50%);
  width:38px;height:38px;border-radius:50%;border:1px solid var(--b2);
  cursor:pointer;background:rgba(0,0,0,.5);backdrop-filter:blur(6px);
  color:#fff;display:flex;align-items:center;justify-content:center;
  transition:all .15s;z-index:2;
}
.lb-nav:hover{background:var(--ac);border-color:transparent}
.lb-nav--l{left:12px}
.lb-nav--r{right:12px}
.lb-counter{position:absolute;bottom:12px;left:50%;transform:translateX(-50%);font-family:var(--mono);font-size:9px;color:rgba(255,255,255,.35);background:rgba(0,0,0,.5);padding:2px 10px;border-radius:20px}
.lb-hint{position:absolute;bottom:12px;right:14px;font-family:var(--mono);font-size:7.5px;color:rgba(255,255,255,.15);letter-spacing:.05em}
.lb-right{border-left:1px solid var(--b1);background:var(--bg1);display:flex;flex-direction:column;min-height:0}
.lb-right-head{display:flex;align-items:center;justify-content:space-between;padding:14px 16px;border-bottom:1px solid var(--b1);flex-shrink:0}
.lb-right-title{font-size:13px;font-weight:600;letter-spacing:-.01em}
.lb-head-btns{display:flex;align-items:center;gap:3px}
.lb-hbtn{background:none;border:none;cursor:pointer;color:var(--t3);padding:6px;border-radius:6px;display:flex;align-items:center;transition:all .12s}
.lb-hbtn:hover{background:var(--bg3);color:var(--t2)}
.lb-close:hover{color:var(--red)}
.lb-scroll{flex:1;overflow-y:auto;padding:12px 14px;display:flex;flex-direction:column;gap:7px}

/* Detail blocks */
.db{background:var(--bg2);border:1px solid var(--b1);border-radius:var(--r2);overflow:hidden;border-left:2px solid var(--dc,var(--ac))}
.db-head{display:flex;align-items:center;gap:6px;padding:7px 11px;border-bottom:1px solid var(--b1)}
.db-label{font-family:var(--mono);font-size:8px;color:var(--dc,var(--ac-l));letter-spacing:.08em;font-weight:500;text-transform:uppercase}
.db-body{padding:9px 11px}
.db-caption{font-size:12px;line-height:1.55;color:var(--t1)}
.db-caption-sub{font-size:11px;line-height:1.55;color:var(--t2);margin-top:4px}
.q-wrap{display:flex;align-items:flex-start;gap:10px}
.q-circle{width:42px;height:42px;border-radius:50%;border:2px solid;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.q-right{flex:1}
.q-level{font-size:12px;font-weight:600;margin-bottom:6px}
.qbar-row{display:flex;align-items:center;gap:7px;margin-bottom:4px}
.qbar-label{font-family:var(--mono);font-size:7.5px;color:var(--t3);width:65px;flex-shrink:0}
.qbar-track{flex:1;height:2px;background:rgba(255,255,255,.07);border-radius:2px;overflow:hidden}
.qbar-fill{height:100%;border-radius:2px}
.qbar-num{font-family:var(--mono);font-size:7.5px;color:var(--t3);width:22px;text-align:right}
.aes-row{display:flex;align-items:center;gap:9px}
.aes-score{font-size:24px;font-weight:800;color:var(--gold)}
.aes-rating{font-size:11px;color:var(--t2)}
.emo-row{display:flex;align-items:center;gap:10px}
.emo-name{font-size:12px;font-weight:600;text-transform:capitalize}
.emo-sub{font-family:var(--mono);font-size:8px;color:var(--t3);margin-top:2px}
.ocr-text{font-family:var(--mono);font-size:9px;color:var(--t2);line-height:1.6;white-space:pre-wrap;max-height:84px;overflow-y:auto}
.tag-wrap{display:flex;flex-wrap:wrap;gap:4px}
.obj-tag{font-family:var(--mono);font-size:8.5px;background:var(--bg3);color:var(--t3);padding:2px 7px;border-radius:4px;border:1px solid var(--b1)}
.meta-stack{display:flex;flex-direction:column;gap:5px}
.mrow{display:flex;justify-content:space-between;align-items:baseline;gap:12px}
.mrow-k{font-family:var(--mono);font-size:8px;color:var(--t3);flex-shrink:0}
.mrow-v{font-size:11px;color:var(--t1);text-align:right}

/* Tags */
.tag-chips{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px}
.tag-chip-item{display:flex;align-items:center;gap:3px;background:rgba(34,197,94,.07);border:1px solid rgba(34,197,94,.15);color:var(--green);font-family:var(--mono);font-size:8.5px;padding:2px 7px;border-radius:4px}
.tag-chip-x{background:none;border:none;cursor:pointer;color:rgba(34,197,94,.4);display:flex;padding:0;margin-left:2px;transition:color .12s}
.tag-chip-x:hover{color:var(--green)}
.tag-add-row{display:flex;gap:4px;align-items:center}
.tag-add-input{flex:1;background:var(--bg3);border:1px solid var(--b1);border-radius:6px;padding:5px 9px;font-family:var(--mono);font-size:10px;color:var(--t1);outline:none;transition:border-color .12s}
.tag-add-input:focus{border-color:rgba(99,102,241,.4)}
.tag-add-btn{background:var(--bg3);border:1px solid var(--b1);border-radius:6px;padding:5px 8px;cursor:pointer;color:var(--t3);display:flex;transition:all .12s}
.tag-add-btn:hover{border-color:rgba(99,102,241,.3);color:var(--ac-l)}

/* ═══ TRASH ═══════════════════════════════════════════════════════════════ */
.trash-card{position:relative;aspect-ratio:1;border-radius:var(--r2);overflow:hidden;background:var(--bg2);border:1px solid var(--b1)}
.trash-img{width:100%;height:100%;object-fit:cover;filter:grayscale(.5) brightness(.4)}
.trash-overlay{position:absolute;inset:0;opacity:0;transition:opacity .18s;background:rgba(0,0,0,.7);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:6px;padding:10px}
.trash-card:hover .trash-overlay{opacity:1}
.trash-btn{width:100%;display:flex;align-items:center;justify-content:center;gap:5px;border:none;border-radius:var(--r);padding:7px;font-family:var(--font);font-size:11px;font-weight:600;cursor:pointer;transition:all .12s}
.trash-btn--restore{background:var(--ac);color:#fff}
.trash-btn--del{background:rgba(220,50,50,.85);color:#fff}
.trash-date{position:absolute;top:7px;left:7px;font-family:var(--mono);font-size:7.5px;background:rgba(0,0,0,.7);color:var(--t3);padding:2px 6px;border-radius:4px}

/* ═══ PEOPLE ══════════════════════════════════════════════════════════════ */
.people-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(108px,1fr));gap:18px}
.person-card{display:flex;flex-direction:column;align-items:center;gap:8px;cursor:pointer}
.person-avatar{width:68px;height:68px;border-radius:50%;overflow:hidden;border:1px solid var(--b1);background:var(--bg2);display:flex;align-items:center;justify-content:center;transition:border-color .15s}
.person-avatar img{width:100%;height:100%;object-fit:cover}
.person-card:hover .person-avatar{border-color:var(--ac)}
.person-name{font-size:12px;font-weight:600;text-align:center}
.person-count{font-family:var(--mono);font-size:8px;color:var(--t3)}

/* ═══ ALBUMS ══════════════════════════════════════════════════════════════ */
.albums-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px}
.album-card{
  position:relative;aspect-ratio:16/10;
  border-radius:var(--r3);overflow:hidden;cursor:pointer;
  background:var(--bg2);border:1px solid var(--b1);
  transition:border-color .15s;
}
.album-card:hover{border-color:var(--b2)}
.album-cover{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;transition:transform .3s}
.album-card:hover .album-cover{transform:scale(1.04)}
.album-cover-ph{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;color:var(--t3)}
.album-strip{position:absolute;bottom:0;left:0;right:0;height:28px;display:flex;opacity:0;transition:opacity .18s;z-index:2}
.album-card:hover .album-strip{opacity:1}
.album-strip-img{flex:1;height:100%;object-fit:cover;border-right:1px solid rgba(0,0,0,.3)}
.album-info{position:absolute;inset:0;z-index:1;background:linear-gradient(to top,rgba(0,0,0,.9) 0%,transparent 55%);padding:12px;display:flex;flex-direction:column;justify-content:flex-end}
.album-title{font-weight:700;font-size:13px;letter-spacing:-.01em;margin-bottom:5px;line-height:1.3;color:#fff}
.album-meta{display:flex;align-items:center;gap:5px;flex-wrap:wrap}
.album-del-btn{position:absolute;top:8px;right:8px;width:24px;height:24px;border-radius:50%;background:rgba(0,0,0,.65);border:1px solid var(--b2);color:var(--t3);display:flex;align-items:center;justify-content:center;cursor:pointer;z-index:3;opacity:0;transition:all .15s}
.album-card:hover .album-del-btn{opacity:1}
.album-del-btn:hover{background:rgba(220,50,50,.85);color:#fff;border-color:transparent}
.tag-sm{font-family:var(--mono);font-size:8px;background:rgba(255,255,255,.08);color:var(--t3);padding:2px 6px;border-radius:20px}
.tag-sm--accent{background:rgba(99,102,241,.15);color:var(--ac-l)}
.tag-sm--manual{background:rgba(245,158,11,.1);color:rgba(245,158,11,.8)}

/* ═══ DUPLICATES ══════════════════════════════════════════════════════════ */
.dupe-list{display:flex;flex-direction:column;gap:14px}
.dupe-group{background:var(--bg2);border:1px solid var(--b1);border-radius:var(--r3);padding:16px}
.dupe-head{display:flex;align-items:center;gap:7px;margin-bottom:14px;flex-wrap:wrap}

/* ═══ MODAL ═══════════════════════════════════════════════════════════════ */
.modal-bg{position:fixed;inset:0;z-index:200;background:rgba(0,0,0,.8);backdrop-filter:blur(10px);display:flex;align-items:center;justify-content:center;padding:18px}
.modal-box{width:100%;max-height:90vh;overflow-y:auto;background:var(--bg1);border:1px solid var(--b1);border-radius:var(--r3);padding:20px;box-shadow:0 16px 50px rgba(0,0,0,.7)}
.modal-head{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:14px;gap:10px}
.modal-title{font-size:18px;font-weight:700;letter-spacing:-.025em}
.modal-sub{font-family:var(--mono);font-size:8px;color:var(--t3);letter-spacing:.06em;margin-top:3px}
.rename-row{display:flex;gap:7px;margin-bottom:14px;align-items:center}
.rename-in{flex:1;background:var(--bg2);border:1px solid var(--b1);border-radius:var(--r);padding:8px 11px;font-family:var(--font);font-size:13px;color:var(--t1);outline:none;transition:border-color .12s}
.rename-in:focus{border-color:rgba(99,102,241,.4)}
.album-desc{background:var(--ac-bg);border:1px solid rgba(99,102,241,.15);border-radius:var(--r);padding:10px 12px;margin-bottom:14px}
.album-desc-lbl{font-family:var(--mono);font-size:8px;color:var(--ac-l);letter-spacing:.08em;display:block;margin-bottom:4px}
.album-desc p{font-size:12px;line-height:1.6}
.create-album-form{display:flex;flex-direction:column;gap:9px}
.form-input{background:var(--bg2);border:1px solid var(--b1);border-radius:var(--r);padding:10px 12px;font-family:var(--font);font-size:13px;color:var(--t1);outline:none;transition:border-color .12s}
.form-input:focus{border-color:rgba(99,102,241,.4)}
.form-textarea{background:var(--bg2);border:1px solid var(--b1);border-radius:var(--r);padding:10px 12px;font-family:var(--font);font-size:13px;color:var(--t1);outline:none;resize:vertical;transition:border-color .12s}
.form-textarea:focus{border-color:rgba(99,102,241,.4)}

.btn-sm{display:inline-flex;align-items:center;gap:5px;background:var(--bg2);border:1px solid var(--b1);color:var(--t2);border-radius:var(--r);padding:6px 12px;font-family:var(--font);font-size:11px;font-weight:500;cursor:pointer;transition:all .12s}
.btn-sm:hover{border-color:var(--b2);color:var(--t1);background:var(--bg3)}
.btn-sm--primary{background:var(--ac);border-color:transparent;color:#fff}
.btn-sm--primary:hover{background:var(--ac-d)}
.btn-sm--accent{background:var(--ac-bg);border-color:rgba(99,102,241,.25);color:var(--ac-l)}
.btn-sm--accent:hover{background:var(--ac);color:#fff;border-color:transparent}
.btn-sm:disabled{opacity:.35;cursor:default}

/* ═══ STATS ═══════════════════════════════════════════════════════════════ */
.stats-wrap{max-width:820px}
.stats-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:14px}
.stat-card{
  background:var(--bg2);border:1px solid var(--b1);
  border-radius:var(--r2);padding:16px 18px;
  border-top:2px solid var(--sc);
  transition:border-color .15s;
}
.stat-card:hover{border-color:var(--b2)}
.stat-lbl{font-family:var(--mono);font-size:8px;color:var(--t3);letter-spacing:.09em;margin-bottom:8px;text-transform:uppercase}
.stat-val{font-size:36px;font-weight:800;letter-spacing:-.04em;line-height:1;color:var(--sc)}
.stats-sec{background:var(--bg2);border:1px solid var(--b1);border-radius:var(--r2);padding:14px;margin-bottom:8px}
.stats-sec-lbl{font-family:var(--mono);font-size:8px;color:var(--t3);letter-spacing:.09em;margin-bottom:12px;text-transform:uppercase}
.clr-dist{display:flex;flex-wrap:wrap;gap:6px}
.clr-dist-item{display:flex;align-items:center;gap:6px;background:var(--bg3);border:1px solid var(--b1);border-radius:6px;padding:5px 10px;font-size:11px}
.clr-dist-dot{width:10px;height:10px;border-radius:50%;border:1px solid rgba(255,255,255,.1);flex-shrink:0}
.clr-dist-n{font-family:var(--mono);font-size:9px;color:var(--t3)}
.tag-cloud{display:flex;flex-wrap:wrap;gap:5px}
.tag-cloud-item{font-family:var(--mono);font-size:9px;background:var(--ac-bg);color:var(--ac-l);border:1px solid rgba(99,102,241,.15);padding:3px 9px;border-radius:20px}
.tag-n{opacity:.5}

/* ═══ EMPTY / LOADER ══════════════════════════════════════════════════════ */
.empty{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:240px;background:var(--bg2);border:1px dashed var(--b1);border-radius:var(--r3);padding:40px;gap:10px;text-align:center}
.empty--click{cursor:pointer;transition:background .12s}
.empty--click:hover{background:var(--bg3)}
.empty-msg{font-size:16px;font-weight:700;letter-spacing:-.02em}
.empty-sub{font-size:12px;color:var(--t2);max-width:240px;line-height:1.6}
.empty-cta{font-family:var(--mono);font-size:9px;color:var(--ac-l);letter-spacing:.09em;margin-top:4px;text-transform:uppercase}
.loader-wrap{display:flex;flex-direction:column;align-items:center;justify-content:center;height:55vh;gap:12px}
.loader-ring{width:34px;height:34px;border-radius:50%;border:2px solid rgba(99,102,241,.12);border-top-color:var(--ac);animation:spin .7s linear infinite}
.loader-text{font-family:var(--mono);font-size:10px;color:var(--t3);letter-spacing:.1em}
@keyframes spin{to{transform:rotate(360deg)}}
.spin{animation:spin .7s linear infinite}

/* ═══ MISC ════════════════════════════════════════════════════════════════ */
.person-chip{display:flex;align-items:center;padding:5px 11px;background:var(--bg2);border:1px solid var(--b1);border-radius:20px;font-size:12px;color:var(--t3);cursor:pointer;transition:all .12s}
.person-chip:hover{border-color:rgba(99,102,241,.3);color:var(--t2)}
.person-chip--on{background:rgba(49,46,129,.4);border-color:var(--ac);color:var(--ac-l)}

@media(max-width:720px){
  .lb-box{grid-template-columns:1fr}
  .lb-right{border-left:none;border-top:1px solid var(--b1);max-height:40vh}
  .stats-cards{grid-template-columns:repeat(2,1fr)}
  .sidebar{width:180px}
}
`}</style>;
}