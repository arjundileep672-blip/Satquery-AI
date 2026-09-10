import React, { useRef, useState } from 'react';
import { Upload, X, RefreshCw, FileImage } from 'lucide-react';

interface ImageSlotProps {
  label: string;
  sublabel: string;
  file: File | null;
  previewUrl: string | null;
  dimensions: { width: number; height: number } | null;
  isRequired?: boolean;
  isLoading: boolean;
  onFileSelect: (file: File) => void;
  onRemove: () => void;
}

const ImageSlot: React.FC<ImageSlotProps> = ({
  label,
  sublabel,
  file,
  previewUrl,
  dimensions,
  isRequired = false,
  isLoading,
  onFileSelect,
  onRemove,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (!isLoading) setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (isLoading) return;
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFileSelect(e.target.files[0]);
    }
  };

  return (
    <div className="flex-1 border border-white/10 p-3 flex flex-col justify-between">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <FileImage className="w-4 h-4 text-emerald-400" />
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-200">
            {label}
            {isRequired ? (
              <span className="text-emerald-400 ml-1.5 normal-case font-normal text-[11px]">
                (Primary)
              </span>
            ) : (
              <span className="text-slate-500 ml-1.5 normal-case font-normal text-[11px]">
                (Change Detection)
              </span>
            )}
          </h4>
        </div>
        <span className="text-[10px] font-mono text-slate-500">{sublabel}</span>
      </div>

      {file && previewUrl ? (
        <div className="relative group rounded-lg overflow-hidden border border-slate-700/80 bg-slate-950 flex flex-col">
          <div className="h-28 w-full flex items-center justify-center sq-grid-bg p-2 relative">
            <img
              src={previewUrl}
              alt={file.name}
              className="max-h-full max-w-full object-contain rounded shadow"
            />
            <div className="absolute inset-0 bg-slate-950/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-2">
              <button
                type="button"
                onClick={() => !isLoading && fileInputRef.current?.click()}
                disabled={isLoading}
                className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs rounded border border-slate-600 flex items-center gap-1 shadow"
              >
                <RefreshCw className="w-3 h-3" />
                <span>Replace</span>
              </button>
              <button
                type="button"
                onClick={onRemove}
                disabled={isLoading}
                className="px-2.5 py-1 bg-red-900/80 hover:bg-red-800 text-red-200 text-xs rounded border border-red-700 flex items-center gap-1 shadow"
              >
                <X className="w-3 h-3" />
                <span>Remove</span>
              </button>
            </div>
          </div>

          <div className="p-2 bg-slate-900 border-t border-slate-800 flex items-center justify-between text-[11px]">
            <div className="truncate max-w-[180px] font-mono text-slate-300" title={file.name}>
              {file.name}
            </div>
            <div className="text-[10px] font-mono text-slate-400 shrink-0">
              {dimensions ? `${dimensions.width}×${dimensions.height} px` : `${(file.size / 1024).toFixed(0)} KB`}
            </div>
          </div>
        </div>
      ) : (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => !isLoading && fileInputRef.current?.click()}
          className={`h-28 border border-dashed p-3 text-center cursor-pointer transition-all flex flex-col items-center justify-center ${
            isDragging
              ? 'border-emerald-400 bg-emerald-950/20'
              : 'border-slate-700/70 hover:border-slate-600 bg-slate-950/40'
          }`}
        >
          <Upload className="w-6 h-6 text-slate-400 mb-1.5" />
          <p className="text-xs text-slate-300 font-medium">
            Drag & drop or <span className="text-emerald-400 underline">browse</span>
          </p>
          <p className="text-[10px] text-slate-500 mt-1">PNG · JPG · JPEG · WebP · TIFF · GeoTIFF</p>
        </div>
      )}

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*,.png,.jpg,.jpeg,.tif,.tiff,.webp"
        onChange={handleFileChange}
        disabled={isLoading}
        className="hidden"
      />
    </div>
  );
};

interface ImageUploaderProps {
  image1: File | null;
  image2: File | null;
  previewUrl1: string | null;
  previewUrl2: string | null;
  dimensions1: { width: number; height: number } | null;
  dimensions2: { width: number; height: number } | null;
  isLoading: boolean;
  onSelectImage1: (file: File) => void;
  onSelectImage2: (file: File) => void;
  onRemoveImage1: () => void;
  onRemoveImage2: () => void;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({
  image1,
  image2,
  previewUrl1,
  previewUrl2,
  dimensions1,
  dimensions2,
  isLoading,
  onSelectImage1,
  onSelectImage2,
  onRemoveImage1,
  onRemoveImage2,
}) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <ImageSlot
        label="Upload Image 1"
        sublabel="Baseline / T1 Scene"
        file={image1}
        previewUrl={previewUrl1}
        dimensions={dimensions1}
        isRequired={true}
        isLoading={isLoading}
        onFileSelect={onSelectImage1}
        onRemove={onRemoveImage1}
      />

      <ImageSlot
        label="Upload Image 2"
        sublabel="Target / T2 Scene (Optional)"
        file={image2}
        previewUrl={previewUrl2}
        dimensions={dimensions2}
        isRequired={false}
        isLoading={isLoading}
        onFileSelect={onSelectImage2}
        onRemove={onRemoveImage2}
      />
    </div>
  );
};
