// WebGPU's usage and map-mode flags, by their values in the specification.
// TypeScript's DOM library declares the GPU interfaces but not these
// namespaces, and a types package is a dependency to ask for; the values are
// the specification's and cannot change.

export const BufferUsage = {
  MAP_READ: 0x0001,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
} as const;

export const TextureUsage = {
  COPY_SRC: 0x01,
  COPY_DST: 0x02,
  TEXTURE_BINDING: 0x04,
  RENDER_ATTACHMENT: 0x10,
} as const;

export const MapMode = { READ: 0x0001 } as const;
