export interface Chunk {
    id: string,
    text: string,
    type: 'parent' | 'child',
    parentId?: string,
    metaData?: Record<string, any>
}