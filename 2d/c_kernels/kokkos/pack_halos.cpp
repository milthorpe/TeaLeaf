#include "kokkos_shared.hpp"
#include "../../shared.h"

using namespace Kokkos;

// Packs the top halo buffer(s)
void pack_top(
        const int x, const int y, const int halo_depth, 
        KView buffer, KView field, const int depth)
{
    parallel_for(x*y, KOKKOS_LAMBDA (const int index)
    {
         int lines = index/(x - 2*halo_depth);
         int offset = x*(y-halo_depth-depth) + lines*2*halo_depth;
         buffer(index) = field(offset+index);
    });
}

// Packs the bottom halo buffer(s)
void pack_bottom(
        const int x, const int y, const int halo_depth, 
        KView buffer, KView field, const int depth)
{
    parallel_for(x*y, KOKKOS_LAMBDA (const int index)
    {
        int lines = index/(x - 2*halo_depth);
        int offset = x*halo_depth + lines*2*halo_depth;
        buffer(index) = field(offset+index);
    });
}

// Unpacks the top halo buffer(s)
void unpack_top(
        const int x, const int y, const int halo_depth, 
        KView buffer, KView field, const int depth)
{
    parallel_for(x*y, KOKKOS_LAMBDA (const int index)
    {
        int lines = index/(x - 2*halo_depth);
        int offset = x*(y-halo_depth) + lines*2*halo_depth;
        field(offset+index)=buffer(index);
    });
}

// Unpacks the bottom halo buffer(s)
void unpack_bottom(
        const int x, const int y, const int halo_depth, 
        KView buffer, KView field, const int depth)
{
    parallel_for(x*y, KOKKOS_LAMBDA (const int index)
    {
        int lines = index/(x - 2*halo_depth);
        int offset = x*(halo_depth-depth) + lines*2*halo_depth;
        field(offset+index)=buffer(index);
    });
}

// Packs the left halo buffer(s)
void pack_left(
        const int x, const int y, const int halo_depth, 
        KView buffer, KView field, const int depth)
{
    parallel_for(x*y, KOKKOS_LAMBDA (const int index)
    {
        int lines = index/depth;
        int offset = halo_depth + lines*(x-depth);
        buffer(index) = field(offset+index);
    });
}

// Packs the right halo buffer(s)
void pack_right(
        const int x, const int y, const int halo_depth, 
        KView buffer, KView field, const int depth)
{
    parallel_for(x*y, KOKKOS_LAMBDA (const int index)
    {        
        int lines = index/depth;
        int offset = x-halo_depth-depth + lines*(x-depth);
        buffer(index) = field(offset+index);
    });
}

// Unpacks the left halo buffer(s)
void unpack_left(
        const int x, const int y, const int halo_depth, 
        KView buffer, KView field, const int depth)
{
    parallel_for(x*y, KOKKOS_LAMBDA (const int index)
    {      
        int lines = index/depth;
        int offset = halo_depth-depth + lines*(x-depth);
        field(offset+index)=buffer(index);
    });
}

// Unpacks the right halo buffer(s)
void unpack_right(
        const int x, const int y, const int halo_depth, 
        KView buffer, KView field, const int depth)
{
    parallel_for(x*y, KOKKOS_LAMBDA (const int index)
    { 
        int lines = index/depth;
        int offset = x-halo_depth + lines*(x-depth);
        field(offset+index)=buffer(index);
    });
}

