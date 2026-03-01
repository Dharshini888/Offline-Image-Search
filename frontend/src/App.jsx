import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Plus, Search, Mic, Image as ImageIcon, Clock, BookImage,
    Users, Map as MapIcon, Heart, Copy, Trash2, Bell, X,
    Sparkles, ImageOff, Info, Calendar
} from 'lucide-react';
import axios from 'axios';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

const API_URL = "http://localhost:8000";

const App = () => {
    const [view, setView] = useState("timeline");
    const [results, setResults] = useState([]);
    const [faces, setFaces] = useState([]);
    const [albums, setAlbums] = useState([]);
    const [loading, setLoading] = useState(false);
    const [reclustering, setReclustering] = useState(false);
    const [reclusterMsg, setReclusterMsg] = useState("");
    const [query, setQuery] = useState("");
    const [selectedImage, setSelectedImage] = useState(null);
    const [selectedPerson, setSelectedPerson] = useState(null);
    const [selectedAlbum, setSelectedAlbum] = useState(null);
    const [personNameEdit, setPersonNameEdit] = useState("");
    const [personSuggestions, setPersonSuggestions] = useState([]);
    const fileInputRef = useRef(null);

    useEffect(() => {
        fetchContent();
    }, [view]);

    const fetchContent = async () => {
        setLoading(true);
        try {
            if (view === "timeline") {
                const res = await axios.get(`${API_URL}/timeline`);
                setResults(res.data.results);
            } else if (view === "faces") {
                const res = await axios.get(`${API_URL}/faces`);
                setFaces(res.data.results);
            } else if (view === "albums") {
                const res = await axios.get(`${API_URL}/albums`);
                setAlbums(res.data.results);
            } else if (view === "favorites") {
                const res = await axios.get(`${API_URL}/favorites`);
                setResults(res.data.results);
            } else if (view === "duplicates") {
                const res = await axios.get(`${API_URL}/duplicates`);
                setResults(res.data.duplicate_groups || []);
            }
        } catch (err) {
            console.error("Fetch failed", err);
        } finally {
            setLoading(false);
        }
    };

    const handleAlbumClick = async (album) => {
        // Set basic info immediately so modal opens, then load photos
        setSelectedAlbum({ ...album, images: null });
        try {
            const res = await axios.get(`${API_URL}/albums/${album.id}`);
            const data = res.data;
            setSelectedAlbum({
                id:          data.id,
                title:       data.title,
                type:        data.type,
                date:        data.date,
                cover:       data.cover,
                image_count: data.image_count,
                images:      data.images || data.results || [],
            });
        } catch (err) {
            console.error("Failed to load album", err);
        }
    };

    const handleSearch = async (e) => {
        if (e) e.preventDefault();
        if (!query.trim()) return;
        setLoading(true);
        setView("search");
        try {
            const formData = new FormData();
            formData.append("query", query);
            const res = await axios.post(`${API_URL}/search`, formData);

            // Handle both success and error responses
            if (res.data.status === "error") {
                setResults([]);
                console.error("Search error:", res.data.message);
            } else if (res.data.results) {
                setResults(res.data.results);
            } else {
                setResults([]);
            }
        } catch (err) {
            console.error("Search failed", err);
            setResults([]);
        } finally {
            setLoading(false);
        }
    };

    const handleUpload = async (files) => {
        setLoading(true);
        try {
            for (let file of files) {
                const formData = new FormData();
                formData.append("file", file);
                await axios.post(`${API_URL}/upload`, formData);
            }
            fetchContent();
        } catch (err) {
            console.error("Upload failed", err);
        } finally {
            setLoading(false);
        }
    };

    const handleFileSelect = (e) => {
        const files = Array.from(e.target.files);
        if (files.length > 0) {
            handleUpload(files);
        }
    };

    const handleRecluster = async () => {
        setReclustering(true);
        setReclusterMsg("Clustering...");
        try {
            const res = await axios.post(`${API_URL}/recluster`);
            const { people, albums: albumCount } = res.data;
            setReclusterMsg(`Done! Found ${people} people, ${albumCount} albums.`);
            // Refresh current view
            fetchContent();
        } catch (err) {
            setReclusterMsg("Clustering failed. Check console.");
            console.error("Recluster failed", err);
        } finally {
            setReclustering(false);
        }
    };

    const toggleFavorite = async (imageId, e) => {
        e.stopPropagation();
        try {
            const formData = new FormData();
            formData.append("image_id", imageId);
            await axios.post(`${API_URL}/favorites`, formData);
            // Refresh current view to update favorite status
            if (view === "favorites" || view === "search" || view === "timeline") {
                fetchContent();
            }
        } catch (err) {
            console.error("Toggle favorite failed", err);
        }
    };

    const handlePersonClick = async (person) => {
        setPersonSuggestions([]);
        setPersonNameEdit(person.name);
        // Set basic info immediately so modal opens instantly
        setSelectedPerson({ ...person, images: null });
        try {
            // Fetch full person data including their photo gallery
            const res = await axios.get(`${API_URL}/people/${person.id}`);
            const data = res.data;
            setSelectedPerson({
                id:         data.id,
                name:       data.name,
                count:      data.face_count ?? data.images?.length ?? person.count,
                cover:      data.cover,
                images:     data.images || data.results || [],
            });
        } catch (err) {
            console.error("Failed to load person detail", err);
        }
    };

    const handleRenamePerson = async (personId, newName) => {
        try {
            const formData = new FormData();
            formData.append("name", newName);
            await axios.post(`${API_URL}/people/${personId}`, formData);
            setPersonNameEdit("");
            fetchContent(); // Refresh faces list
            if (selectedPerson) {
                setSelectedPerson({ ...selectedPerson, name: newName });
            }
        } catch (err) {
            console.error("Rename failed", err);
        }
    };

    const handleCelebCheck = async (personId) => {
        try {
            const res = await axios.get(`${API_URL}/people/${personId}/celebcheck`);
            if (res.data.status === "suggestions") {
                setPersonSuggestions(res.data.suggestions || []);
            } else {
                setPersonSuggestions([res.data.message || "No suggestions found"]);
            }
        } catch (err) {
            console.error("Celebrity check failed", err);
            setPersonSuggestions(["Error checking celebrity"]);
        }
    };

    return (
        <div
            className="flex h-screen overflow-hidden bg-[#09090b] text-[#fafafa]"
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
                e.preventDefault();
                const files = Array.from(e.dataTransfer.files);
                handleUpload(files);
            }}
        >
            {/* Sidebar */}
            <aside className="w-64 border-r border-white/5 flex flex-col p-6 z-20">
                <div className="flex items-center gap-3 mb-12">
                    <div className="w-10 h-10 bg-blue-600 rounded-2xl flex items-center justify-center shadow-lg shadow-blue-500/20">
                        <Sparkles className="text-white w-6 h-6" />
                    </div>
                    <h1 className="font-bold text-lg tracking-tight">SmartGallery</h1>
                </div>

                <nav className="space-y-1">
                    <NavLink active={view === "timeline"} onClick={() => setView("timeline")} icon={<Clock size={20} />} label="Timeline" />
                    <NavLink active={view === "albums"} onClick={() => setView("albums")} icon={<BookImage size={20} />} label="Albums" />
                    <NavLink active={view === "faces"} onClick={() => setView("faces")} icon={<Users size={20} />} label="People" />
                    <NavLink active={view === "map"} onClick={() => setView("map")} icon={<MapIcon size={20} />} label="Map" />
                    <div className="pt-6 pb-2 text-[10px] font-bold text-zinc-500 tracking-widest uppercase ml-4">Library</div>
                    <NavLink active={view === "favorites"} onClick={() => setView("favorites")} icon={<Heart size={20} />} label="Favorites" />
                    <NavLink active={view === "duplicates"} onClick={() => setView("duplicates")} icon={<Copy size={20} />} label="Duplicates" />
                    <NavLink active={false} icon={<Trash2 size={20} />} label="Trash" />
                </nav>

                <div className="mt-auto space-y-3">
                    {/* Recluster Button */}
                    <button
                        onClick={handleRecluster}
                        disabled={reclustering}
                        className="w-full flex items-center justify-center gap-2 bg-white/5 hover:bg-white/10 border border-white/5 text-zinc-300 px-3 py-2 rounded-xl text-xs font-semibold transition-all disabled:opacity-50"
                    >
                        <Sparkles size={14} className={reclustering ? "animate-spin text-blue-400" : "text-zinc-500"} />
                        {reclustering ? "Indexing..." : "Re-index People & Albums"}
                    </button>
                    {reclusterMsg && (
                        <p className="text-[10px] text-center text-zinc-500">{reclusterMsg}</p>
                    )}
                    <div className="bg-white/5 p-4 rounded-2xl border border-white/5">
                        <div className="flex items-center gap-3 mb-3">
                            <div className="w-2 h-2 rounded-full bg-green-500"></div>
                            <span className="text-xs font-medium">Fully Offline</span>
                        </div>
                        <div className="text-[10px] text-zinc-500">
                            AI processing active. <br /> CLIP + FaceNet enabled.
                        </div>
                    </div>
                </div>
            </aside>

            {/* Main Content */}
            <main className="flex-1 flex flex-col overflow-hidden">
                <header className="p-6 flex items-center justify-between border-b border-white/5">
                    <form onSubmit={handleSearch} className="relative w-full max-w-xl group">
                        <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500 group-focus-within:text-blue-500 transition-colors" />
                        <input
                            type="text"
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            placeholder="Search for 'vacation 2024' or '🐶 at 🏖️'..."
                            className="w-full bg-zinc-900 border border-white/5 rounded-2xl py-3 pl-12 pr-4 focus:outline-none focus:ring-1 focus:ring-blue-500/50 transition-all font-medium"
                        />
                        <button
                            type="button"
                            title="Voice search"
                            onClick={async () => {
                                setLoading(true);
                                setView("search");
                                try {
                                    const formData = new FormData();
                                    formData.append("duration", "5");
                                    const res = await axios.post(`${API_URL}/search/voice`, formData);
                                    if (res.data.transcribed) setQuery(res.data.transcribed);
                                    if (res.data.results) setResults(res.data.results);
                                    else setResults([]);
                                } catch (err) {
                                    console.error("Voice search failed", err);
                                } finally {
                                    setLoading(false);
                                }
                            }}
                            className="absolute right-4 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-blue-400 transition-colors"
                        >
                            <Mic className="w-4 h-4" />
                        </button>
                    </form>
                    <div className="flex items-center gap-3">
                        <button
                            onClick={() => fileInputRef.current?.click()}
                            className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-xl text-sm font-bold shadow-lg shadow-blue-500/20 transition-all hover:scale-105 active:scale-95"
                        >
                            <Plus size={18} />
                            <span className="hidden md:inline">Add Photos</span>
                        </button>
                        <button className="p-2 hover:bg-white/5 rounded-xl transition-colors text-zinc-400"><Bell className="w-5 h-5" /></button>
                        <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-blue-500 to-purple-500 border-2 border-white/10 shadow-lg"></div>
                    </div>
                </header>

                <div className="flex-1 overflow-y-auto p-6 md:p-8 custom-scrollbar">
                    <AnimatePresence mode="wait">
                        {loading ? (
                            <motion.div key="loader" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col items-center justify-center h-full">
                                <div className="w-10 h-10 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                                <p className="mt-4 text-sm text-zinc-500">Processing images...</p>
                            </motion.div>
                        ) : (
                            <motion.div key={view} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="max-w-7xl mx-auto">
                                {view === "timeline" || view === "search" ? (
                                    <div>
                                        <div className="flex items-center justify-between mb-8">
                                            <h2 className="text-2xl font-bold tracking-tight">
                                                {view === "search" ? `Results for "${query}"` : "Recent Photos"}
                                            </h2>
                                        </div>
                                        {results.length === 0 ? (
                                            <div
                                                onClick={view !== "search" ? () => fileInputRef.current?.click() : undefined}
                                                className={`flex flex-col items-center justify-center py-20 text-zinc-500 ${view !== "search" ? "cursor-pointer hover:bg-white/5" : ""} rounded-3xl border-2 border-dashed border-white/5 transition-all group`}
                                            >
                                                {view === "search" ? (
                                                    <>
                                                        <ImageOff className="w-16 h-16 mb-6 opacity-30 group-hover:opacity-50 transition-opacity" />
                                                        <p className="text-xl font-bold text-white mb-2">No results found</p>
                                                        <p className="text-sm">Try different keywords like "beach", "dog", or "sunset". Make sure to:</p>
                                                        <ul className="text-sm list-disc list-inside mt-3 text-zinc-400">
                                                            <li>Upload images first</li>
                                                            <li>Click "Re-index" button to process images</li>
                                                            <li>Use specific keywords</li>
                                                        </ul>
                                                    </>
                                                ) : (
                                                    <>
                                                        <div className="w-16 h-16 bg-blue-600/10 rounded-full flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                                                            <Plus className="w-8 h-8 text-blue-500" />
                                                        </div>
                                                        <p className="text-xl font-bold text-white mb-2">Ready to add photos?</p>
                                                        <p className="text-sm">Click here to open file explorer or drag and drop images anywhere</p>
                                                    </>
                                                )}
                                            </div>
                                        ) : (
                                            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
                                                {results.map((img, i) => (
                                                    <ImageCard key={i} image={img} onClick={() => setSelectedImage(img)} onFavorite={toggleFavorite} />
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                ) : view === "faces" ? (
                                    <div>
                                        <div className="flex items-center justify-between mb-8">
                                            <h2 className="text-2xl font-bold tracking-tight">People</h2>
                                        </div>
                                        {faces.length === 0 ? (
                                            <div className="flex flex-col items-center justify-center py-20 text-zinc-500 bg-white/5 rounded-3xl border border-white/5">
                                                <Users className="w-12 h-12 mb-4 opacity-20" />
                                                <p className="text-lg font-bold text-white mb-2">No people identified yet</p>
                                                <p className="text-sm">Run 'build_index.py' or upload photos with clear faces.</p>
                                            </div>
                                        ) : (
                                            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-6">
                                                {faces.map((p, i) => (
                                                    <FaceCircle key={i} person={p} onClick={() => handlePersonClick(p)} />
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                ) : view === "albums" ? (
                                    <div>
                                        <div className="flex items-center justify-between mb-8">
                                            <h2 className="text-2xl font-bold tracking-tight">Albums</h2>
                                        </div>
                                        {albums.length === 0 ? (
                                            <div className="flex flex-col items-center justify-center py-20 text-zinc-500 bg-white/5 rounded-3xl border border-white/5">
                                                <BookImage className="w-12 h-12 mb-4 opacity-20" />
                                                <p className="text-lg font-bold text-white mb-2">No albums created yet</p>
                                                <p className="text-sm">Albums are automatically created for trips and events.</p>
                                            </div>
                                        ) : (
                                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                                                {albums.map((album, i) => (
                                                    <AlbumCard key={i} album={album} onClick={() => handleAlbumClick(album)} />
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                ) : view === "map" ? (
                                    <div className="h-[600px] rounded-3xl overflow-hidden border border-white/5">
                                        <MapContainer center={[20, 0]} zoom={2} style={{ height: '100%', width: '100%', background: '#09090b' }}>
                                            <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                                            {results.filter(img => img.location?.lat).map((img, i) => (
                                                <Marker key={i} position={[img.location.lat, img.location.lon]}>
                                                    <Popup>
                                                        <img src={`${API_URL}/images/${img.filename}`} className="w-32 rounded-lg" />
                                                    </Popup>
                                                </Marker>
                                            ))}
                                        </MapContainer>
                                    </div>
                                ) : view === "favorites" ? (
                                    <div>
                                        <div className="flex items-center justify-between mb-8">
                                            <h2 className="text-2xl font-bold tracking-tight">My Favorites</h2>
                                        </div>
                                        {results.length === 0 ? (
                                            <div className="flex flex-col items-center justify-center py-20 text-zinc-500 bg-white/5 rounded-3xl border border-white/5">
                                                <Heart className="w-12 h-12 mb-4 opacity-20" />
                                                <p className="text-lg font-bold text-white mb-2">No favorites yet</p>
                                                <p className="text-sm">Heart your favorite photos to save them here.</p>
                                            </div>
                                        ) : (
                                            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
                                                {results.map((img, i) => (
                                                    <ImageCard key={i} image={img} onClick={() => setSelectedImage(img)} onFavorite={toggleFavorite} />
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                ) : view === "duplicates" ? (
                                    <div>
                                        <div className="flex items-center justify-between mb-8">
                                            <h2 className="text-2xl font-bold tracking-tight">Duplicate Images</h2>
                                        </div>
                                        {results.length === 0 ? (
                                            <div className="flex flex-col items-center justify-center py-20 text-zinc-500 bg-white/5 rounded-3xl border border-white/5">
                                                <Copy className="w-12 h-12 mb-4 opacity-20" />
                                                <p className="text-lg font-bold text-white mb-2">No duplicates found</p>
                                                <p className="text-sm">Similar images are detected using perceptual hashing.</p>
                                            </div>
                                        ) : (
                                            <div className="space-y-8">
                                                {results.map((group, i) => (
                                                    <div key={i} className="bg-white/5 rounded-2xl border border-white/5 p-6">
                                                        <h3 className="text-sm font-bold text-zinc-300 mb-4">Duplicate Group ({group.count} images)</h3>
                                                        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                                                            {group.images && group.images.map((img, j) => (
                                                                <div key={j} className="relative aspect-square rounded-lg overflow-hidden cursor-pointer group/dup bg-zinc-900 border border-white/5">
                                                                    <img
                                                                        src={`${API_URL}/images/${img.filename}`}
                                                                        className="w-full h-full object-cover group-hover/dup:opacity-75 transition-opacity"
                                                                        onClick={() => setSelectedImage(img)}
                                                                    />
                                                                    <div className="absolute inset-0 bg-black/50 opacity-0 group-hover/dup:opacity-100 transition-opacity flex flex-col items-center justify-center">
                                                                        {img.size && (
                                                                            <p className="text-xs text-zinc-300">{(img.size / 1024).toFixed(1)}KB</p>
                                                                        )}
                                                                    </div>
                                                                </div>
                                                            ))}
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                ) : null}
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            </main>

            {selectedImage && (
                <Lightbox image={selectedImage} onClose={() => setSelectedImage(null)} />
            )}

            {selectedAlbum && (
                <AlbumDetail
                    album={selectedAlbum}
                    onClose={() => setSelectedAlbum(null)}
                />
            )}

            {selectedPerson && (
                <PersonDetail
                    person={selectedPerson}
                    onClose={() => setSelectedPerson(null)}
                    onRename={handleRenamePerson}
                    onCelebCheck={handleCelebCheck}
                    suggestions={personSuggestions}
                    nameEdit={personNameEdit}
                    onNameEditChange={setPersonNameEdit}
                />
            )}

            <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileSelect}
                multiple
                accept="image/*"
                className="hidden"
            />
        </div>
    );
};

const NavLink = ({ active, onClick, icon, label }) => (
    <button
        onClick={onClick}
        className={`w-full flex items-center gap-4 px-4 py-3 rounded-2xl transition-all duration-300 ${active ? 'bg-blue-600/10 text-blue-500' : 'text-zinc-500 hover:bg-white/5 hover:text-white'}`}
    >
        {icon}
        <span className="text-sm font-semibold">{label}</span>
        {active && <motion.div layoutId="nav-pill" className="ml-auto w-1 h-4 bg-blue-500 rounded-full" />}
    </button>
);

const ImageCard = ({ image, onClick, onFavorite }) => (
    <motion.div
        whileHover={{ scale: 1.02, y: -4 }}
        onClick={onClick}
        className="relative aspect-square rounded-2xl overflow-hidden cursor-pointer group bg-zinc-900 border border-white/5 shadow-lg"
    >
        <img
            src={`${API_URL}/images/${image.filename}`}
            className="w-full h-full object-cover"
            loading="lazy"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity p-4 flex flex-col justify-between">
            <button
                onClick={(e) => onFavorite && onFavorite(image.id, e)}
                className="self-end p-2 rounded-full bg-black/50 hover:bg-black/80 transition-colors"
            >
                <Heart size={18} className="text-red-500 fill-red-500" />
            </button>
            <div>
                <p className="text-[10px] text-zinc-400">{new Date(image.timestamp || image.date).toLocaleDateString()}</p>
                {image.score && <p className="text-xs font-bold text-blue-400">{image.score}% Match</p>}
            </div>
        </div>
    </motion.div>
);

const FaceCircle = ({ person, onClick }) => (
    <div onClick={onClick} className="flex flex-col items-center gap-3 group cursor-pointer hover:scale-105 transition-transform">
        <div className="w-24 h-24 rounded-full border-2 border-white/5 group-hover:border-blue-500 transition-colors bg-zinc-900 overflow-hidden shadow-lg">
            {person.cover ? (
                <img
                    src={`${API_URL}${person.cover}`}
                    className="w-full h-full object-cover rounded-full"
                />
            ) : (
                <div className="w-full h-full flex items-center justify-center bg-zinc-800 rounded-full">
                    <Users className="text-zinc-600" />
                </div>
            )}
        </div>
        <div className="text-center">
            <p className="text-sm font-bold text-zinc-200 group-hover:text-blue-400 transition-colors">{person.name}</p>
            <p className="text-[10px] text-zinc-500 font-medium">{person.count} photos</p>
        </div>
    </div>
);

const AlbumCard = ({ album, onClick }) => (
    <motion.div
        whileHover={{ scale: 1.02, y: -4 }}
        onClick={onClick}
        className="relative aspect-[4/3] rounded-2xl overflow-hidden cursor-pointer group bg-zinc-900 border border-white/5 shadow-lg"
    >
        {/* Cover image — API returns /images/filename so use it directly */}
        {album.cover ? (
            <img
                src={`${API_URL}${album.cover}`}
                className="absolute inset-0 w-full h-full object-cover"
                loading="lazy"
                onError={(e) => { e.target.style.display = 'none'; }}
            />
        ) : (
            <div className="absolute inset-0 bg-zinc-800 flex items-center justify-center">
                <BookImage className="w-12 h-12 text-zinc-700" />
            </div>
        )}
        {/* Thumbnail strip */}
        {album.thumbnails && album.thumbnails.length > 1 && (
            <div className="absolute bottom-0 left-0 right-0 flex h-12 gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                {album.thumbnails.slice(0, 4).map((t, i) => (
                    <img key={i} src={`${API_URL}${t}`} className="flex-1 h-full object-cover" />
                ))}
            </div>
        )}
        {/* Overlay */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/30 to-transparent p-5 flex flex-col justify-end">
            <h3 className="text-base font-bold text-white mb-1 truncate">{album.title}</h3>
            <div className="flex items-center gap-2">
                <span className="bg-blue-600/20 text-blue-400 text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider">
                    {album.type || 'Collection'}
                </span>
                <span className="text-zinc-400 text-[10px]">{album.count} photos</span>
                {album.date && <span className="text-zinc-500 text-[10px]">{album.date}</span>}
            </div>
        </div>
    </motion.div>
);

const Lightbox = ({ image, onClose }) => (
    <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center p-4 md:p-12 bg-black/95 backdrop-blur-3xl"
    >
        <button onClick={onClose} className="absolute top-8 right-8 text-zinc-500 hover:text-white transition-colors p-2 z-50">
            <X size={32} />
        </button>
        <div className="max-w-5xl w-full h-full flex flex-col items-center justify-center pointer-events-none">
            <img src={`${API_URL}/images/${image.filename}`} className="max-h-full max-w-full rounded-2xl shadow-2xl object-contain shadow-blue-500/10 pointer-events-auto" />
            <div className="mt-8 flex gap-4 pointer-events-auto">
                <div className="bg-white/5 border border-white/5 px-4 py-2 rounded-xl text-xs flex items-center gap-2">
                    <Calendar size={14} className="text-blue-400" />
                    {new Date(image.timestamp || image.date).toLocaleString()}
                </div>
                <div className="bg-white/5 border border-white/5 px-4 py-2 rounded-xl text-xs flex items-center gap-2">
                    <Info size={14} className="text-blue-400" />
                    {image.filename}
                </div>
            </div>
        </div>
    </motion.div>
);

const PersonDetail = ({ person, onClose, onRename, onCelebCheck, suggestions, nameEdit, onNameEditChange }) => {
    const [isEditing, setIsEditing] = React.useState(false);

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/95 backdrop-blur-3xl"
        >
            <motion.div
                initial={{ scale: 0.95, y: 20 }}
                animate={{ scale: 1, y: 0 }}
                exit={{ scale: 0.95, y: 20 }}
                className="w-full max-w-2xl bg-zinc-900 border border-white/5 rounded-3xl p-8 max-h-[90vh] overflow-y-auto"
            >
                <div className="flex items-center justify-between mb-6">
                    <h2 className="text-2xl font-bold tracking-tight">Person Details</h2>
                    <button onClick={onClose} className="p-2 hover:bg-white/5 rounded-xl transition-colors">
                        <X size={24} className="text-zinc-400" />
                    </button>
                </div>

                {/* Name editing section */}
                <div className="mb-8">
                    <p className="text-xs font-bold text-zinc-500 uppercase tracking-wider mb-2">Name</p>
                    {!isEditing ? (
                        <div className="flex items-center gap-3">
                            <h3 className="text-xl font-bold text-white">{person.name}</h3>
                            <button
                                onClick={() => setIsEditing(true)}
                                className="p-2 hover:bg-white/5 rounded-lg transition-colors text-zinc-400 hover:text-white"
                            >
                                <Copy size={16} />
                            </button>
                        </div>
                    ) : (
                        <div className="flex gap-2 mb-3">
                            <input
                                type="text"
                                value={nameEdit}
                                onChange={(e) => onNameEditChange(e.target.value)}
                                className="flex-1 bg-zinc-800 border border-white/10 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
                                placeholder="Enter person name"
                                autoFocus
                            />
                            <button
                                onClick={() => {
                                    onRename(person.id, nameEdit);
                                    setIsEditing(false);
                                }}
                                className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg font-semibold transition-all"
                            >
                                Save
                            </button>
                            <button
                                onClick={() => setIsEditing(false)}
                                className="bg-white/5 hover:bg-white/10 text-white px-3 py-2 rounded-lg transition-all"
                            >
                                Cancel
                            </button>
                        </div>
                    )}

                    {/* Celebrity check section */}
                    {!isEditing && (
                        <div className="mt-4 flex gap-2">
                            <button
                                onClick={() => onCelebCheck(person.id)}
                                className="text-xs bg-white/5 hover:bg-white/10 border border-white/5 text-zinc-300 px-3 py-1.5 rounded-lg transition-all"
                            >
                                🔍 Auto-identify
                            </button>
                            {suggestions.length > 0 && (
                                <div className="text-xs text-zinc-400 bg-white/5 rounded-lg px-3 py-1.5">
                                    {Array.isArray(suggestions) ? (
                                        <div>
                                            {suggestions.map((s, i) => (
                                                <p key={i} className="text-blue-400 cursor-pointer hover:underline">
                                                    {s}
                                                </p>
                                            ))}
                                        </div>
                                    ) : (
                                        <p>{suggestions}</p>
                                    )}
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* Face count */}
                <div className="mb-8">
                    <p className="text-xs font-bold text-zinc-500 uppercase tracking-wider mb-3">Appearances</p>
                    <p className="text-lg font-semibold text-white">{person.count} photos</p>
                </div>

                {/* Photo gallery */}
                <div className="mb-6">
                    <p className="text-xs font-bold text-zinc-500 uppercase tracking-wider mb-4">Photos</p>
                    {person.images === null ? (
                        <div className="flex items-center justify-center py-8">
                            <div className="w-8 h-8 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
                        </div>
                    ) : person.images && person.images.length > 0 ? (
                        <div className="grid grid-cols-3 gap-4">
                            {person.images.map((img, i) => (
                                <div key={i} className="aspect-square rounded-lg overflow-hidden bg-zinc-800 border border-white/5">
                                    <img
                                        src={`${API_URL}${img.thumbnail ? img.thumbnail : "/images/" + img.filename}`}
                                        className="w-full h-full object-cover hover:scale-110 transition-transform cursor-pointer"
                                        title={img.date}
                                    />
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className="flex flex-col items-center py-8 text-zinc-500">
                            <ImageOff className="w-10 h-10 mb-3 opacity-20" />
                            <p className="text-sm">No photos found for this person.</p>
                            <p className="text-xs mt-1 text-zinc-600">Try clicking Re-index People &amp; Albums.</p>
                        </div>
                    )}
                </div>
            </motion.div>
        </motion.div>
    );
};
const AlbumDetail = ({ album, onClose }) => (
    <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/95 backdrop-blur-3xl"
    >
        <motion.div
            initial={{ scale: 0.95, y: 20 }}
            animate={{ scale: 1, y: 0 }}
            exit={{ scale: 0.95, y: 20 }}
            className="w-full max-w-4xl bg-zinc-900 border border-white/5 rounded-3xl p-8 max-h-[90vh] overflow-y-auto"
        >
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h2 className="text-2xl font-bold tracking-tight">{album.title}</h2>
                    <div className="flex items-center gap-3 mt-1">
                        {album.type && (
                            <span className="bg-blue-600/20 text-blue-400 text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider">
                                {album.type}
                            </span>
                        )}
                        {album.date && <span className="text-xs text-zinc-500">{album.date}</span>}
                        {album.image_count != null && (
                            <span className="text-xs text-zinc-500">{album.image_count} photos</span>
                        )}
                    </div>
                </div>
                <button onClick={onClose} className="p-2 hover:bg-white/5 rounded-xl transition-colors">
                    <X size={24} className="text-zinc-400" />
                </button>
            </div>

            {album.images === null ? (
                <div className="flex items-center justify-center py-20">
                    <div className="w-8 h-8 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
                </div>
            ) : album.images.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-20 text-zinc-500">
                    <ImageOff className="w-12 h-12 mb-4 opacity-20" />
                    <p className="text-lg font-bold text-white mb-2">No photos in this album yet</p>
                    <p className="text-sm">Try clicking Re-index People &amp; Albums to rebuild.</p>
                </div>
            ) : (
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                    {album.images.map((img, i) => (
                        <div key={i} className="aspect-square rounded-xl overflow-hidden bg-zinc-800 border border-white/5 group cursor-pointer">
                            <img
                                src={`${API_URL}${img.thumbnail ? img.thumbnail : "/images/" + img.filename}`}
                                className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                                title={img.date}
                            />
                        </div>
                    ))}
                </div>
            )}
        </motion.div>
    </motion.div>
);

export default App;