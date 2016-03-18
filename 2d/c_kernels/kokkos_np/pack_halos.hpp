#pragma once
#include "kokkos_shared.hpp"
#include "../../shared.h"

// Packs or unpacks halos
template <class Device>
struct HaloPacker
{
    typedef Device device_type;

    HaloPacker(const int x, const int y, const int halo_depth, 
            KView buffer, KView field, const int depth, const int face, 
            const bool pack) 
        : x(x), y(y), halo_depth(halo_depth), buffer(buffer), field(field), 
        depth(depth), face(face), pack(pack){}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int index) const 
    {
        if(face == CHUNK_TOP && pack)
        {
            int lines = index/(x - 2*halo_depth);
            int offset = x*(y-halo_depth-depth) + lines*2*halo_depth;
            buffer(index) = field(offset+index);
        }
        else if(face == CHUNK_BOTTOM && pack)
        {
            int lines = index/(x - 2*halo_depth);
            int offset = x*halo_depth + lines*2*halo_depth;
            buffer(index) = field(offset+index);
        }
        else if (face == CHUNK_TOP && !pack)
        {
            int lines = index/(x - 2*halo_depth);
            int offset = x*(y-halo_depth) + lines*2*halo_depth;
            field(offset+index)=buffer(index);
        }
        else if (face == CHUNK_BOTTOM && !pack)
        {
            int lines = index/(x - 2*halo_depth);
            int offset = x*(halo_depth-depth) + lines*2*halo_depth;
            field(offset+index)=buffer(index);
        }
        else if(face == CHUNK_LEFT && pack)
        {
	        int lines = index/depth;
            int offset = halo_depth + lines*(x-depth);
            buffer(index) = field(offset+index);
        }
        else if(face == CHUNK_RIGHT && pack)
        {
	        int lines = index/depth;
            int offset = x-halo_depth-depth + lines*(x-depth);
            buffer(index) = field(offset+index);
        }
        else if(face == CHUNK_LEFT && !pack)
        {
	        int lines = index/depth;
            int offset = halo_depth-depth + lines*(x-depth);
            field(offset+index)=buffer(index);
        }
        else if(face == CHUNK_RIGHT && !pack)
        {
	        int lines = index/depth;
            int offset = x-halo_depth + lines*(x-depth);
            field(offset+index)=buffer(index);
        }
    }

    const int x;
    const int y;
    const int halo_depth;
    const int depth;
    const int face;
    const bool pack;
    KView buffer;
    KView field;
};

