// SPDX-FileCopyrightText: 2026 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file NeighborCellsAcrossFace.h
 *
 * Companion utilities to class StencilHelper.
 * Implement a function to iterate over all neighbor cells across a face of current cell (when
 * looping over all cell of an AMR mesh).
 *
 */
#ifndef KALYPSSO_CORE_NEIGHBORCELLSACROSSFACE_H_
#define KALYPSSO_CORE_NEIGHBORCELLSACROSSFACE_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>

#include <kalypsso/core/StencilHelper.h>

#include <kalypsso/core/ConformalNeighborStatus.h>
#include <kalypsso/core/ConformalFaceStatus.h>
#include <kalypsso/core/orchard_key_utils.h> // for swapBits
#include <kalypsso/core/enums.h>

namespace kalypsso
{

// ====================================================================================
// ====================================================================================
/**
 * \struct NeighborCellAcrossFaceInfo
 *
 * Context passed as second argument to the functor callback used in
 * for_each_neighbor_cell_across_face, for a given neighbor cell.
 */
template <size_t dim>
struct NeighborCellAcrossFaceInfo
{
  //! relative AMR level of the neighbor with respect to the current cell
  conformal_neighbor_status::neighbor_status status;

  //! 0-based index of this neighbor among num_neighbors (only relevant when status is
  //! NEIGHBOR_IS_FINER)
  uint8_t i_neighbor;

  //! total number of neighbor cells across a given face for this status.
  //! valid values are
  //! - 1 when status is NEIGHBOR_IS_AT_SAME or NEIGHBOR_IS_COARSER,
  //! - 2^(dim-1) when status is NEIGHBOR_IS_FINER
  uint8_t num_neighbors;

}; // struct NeighborCellAcrossFaceInfo

// ====================================================================================
// ====================================================================================
/**
 * Compute the cell location of the n-th finer sibling cell location across a given face.
 *
 * This function assumes:
 * - neighbor cell across given face is at finer AMR level (multiple neighbor cells exists)
 * - the 0-th finer sibling is known as it is given by a call to
 *   StencilHelper::getNeighLocFinerNearer
 *
 * Here we just computes the cell location of the n-th sibling (in the neighbor finer block).
 *
 * \param[in] cell_loc_neigh_finer_nearer must be a cell location obtained from
 *            StencilHelper::getNeighLocFinerNearer
 * \param[in] dir direction orthogonal to the face (IX, IY or IZ)
 * \param[in] i_sibling index in [0, 2^(dim-1)[
 *
 * \note \sa StencilHelper<dim,device_t>::compute_face_siblings_sum
 * \note siblings are enumerated in Morton order, i.e.
 * - i_sibling = 0 corresponds to the siblings with the lowest Morton index
 * - i_sibling = 2^(dim-1)-1 corresponds to the siblings with the largest Morton index
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION auto
get_finer_sibling_cell_loc(CellLocation<dim> const & cell_loc_neigh_finer_nearer,
                           int                       dir,
                           uint16_t                  i_sibling) -> CellLocation<dim>
{
  auto const & coord = cell_loc_neigh_finer_nearer.ijk;

  // round down coordinates to a multiple of two (to get eldest sibling)
  coord_t<dim> coord0;
  coord0[IX] = (coord[IX] / 2) * 2;
  coord0[IY] = (coord[IY] / 2) * 2;
  if constexpr (dim == 3)
    coord0[IZ] = (coord[IZ] / 2) * 2;

  // index ii spans all siblings index matching a face orthogonal to direction "dir"
  uint16_t ii = swapBits(i_sibling, dim - 1, dir);

  // if coord[dir] is odd we are in a right face, if even we are in a left face
  if (coord[static_cast<size_t>(dir)] % 2 == 1)
    ii += static_cast<uint16_t>(1 << dir);

  CellLocation<dim> sibling_loc = cell_loc_neigh_finer_nearer;

  if constexpr (dim == 2)
  {
    sibling_loc.ijk[IX] = coord0[IX] + ((ii & 0x1) >> 0);
    sibling_loc.ijk[IY] = coord0[IY] + ((ii & 0x2) >> 1);
  }
  else if constexpr (dim == 3)
  {
    sibling_loc.ijk[IX] = coord0[IX] + ((ii & 0x1) >> 0);
    sibling_loc.ijk[IY] = coord0[IY] + ((ii & 0x2) >> 1);
    sibling_loc.ijk[IZ] = coord0[IZ] + ((ii & 0x4) >> 2);
  }

  return sibling_loc;

} // get_finer_sibling_cell_loc

// ====================================================================================
// ====================================================================================
/**
 * Visit all neighbor cell(s) across a given face of a cell.
 *
 * - cell is identified by a cell location,
 * - face is identified by a Face::face_t value (e.g. Face::XMIN)
 *
 * While looping over all neighbor cells, a functor is called once per neighbor:
 * - once, if the neighbor is at the same level or coarser
 * - once per finer sibling (2 in 2D, 4 in 3D), if the neighbor is finer
 *
 * Functor should be given in input a cell location identifying one of the neighbor cell(s).
 *
 * This function does NOT perform any aggregation/weighting itself (no sum, no volume ratio) -
 * the functor is entirely responsible for deciding what to do with each neighbor.
 *
 * \note Functor can be a KOKKOS lambda function that will capture all necessary variable from
 * calling context.
 *
 * \param[in] stencil_helper A stencil helper object
 * \param[in] cell_loc current cell location
 * \param[in] current block (octree leaf) conformal status
 * \param[in] face a face id (e.g. Face::XMIN, ....)
 * \param[in] functor a functor that be be called once per neighbor cell
 */
template <size_t dim, typename device_t, typename Func>
KOKKOS_INLINE_FUNCTION void
for_each_neighbor_cell_across_face(StencilHelper<dim, device_t> const & stencil_helper,
                                   CellLocation<dim> const &            cell_loc,
                                   conformal_status_t<dim> const &      conformal_status,
                                   Face::face_t const &                 face,
                                   Func &&                              functor)
{
  using CellLocation_t = CellLocation<dim>;

  const auto status = conformal_face_status_t<dim>::get_status(face, conformal_status);

  KOKKOS_ASSERT((status != conformal_neighbor_status::NEIGHBOR_IS_UNAVAILABLE) &&
                "[for_each_cell_neighbor_across_face] neighbor cell status unavailable");

  const int  dir = face / 2;
  const int  sign = Face::is_left_face(face) ? -1 : 1;
  const auto shift = stencil_helper.unit_shift(sign * (dir + 1));

  if (status == conformal_neighbor_status::NEIGHBOR_IS_FINER)
  {
    const CellLocation_t cell_loc_neigh = stencil_helper.getNeighLocFinerNearer(cell_loc, shift);
    constexpr uint8_t    num_neighbors = 1 << (dim - 1);

    for (uint8_t k = 0; k < num_neighbors; ++k)
    {
      const CellLocation_t cell_loc_neigh_sibling =
        get_finer_sibling_cell_loc<dim>(cell_loc_neigh, dir, k);
      functor(cell_loc_neigh_sibling, NeighborCellAcrossFaceInfo<dim>{ status, k, num_neighbors });
    }
  }
  else
  {
    const CellLocation_t cell_loc_neigh = stencil_helper.getNeighLoc(cell_loc, shift);
    functor(cell_loc_neigh, NeighborCellAcrossFaceInfo<dim>{ status, uint8_t{ 0 }, uint8_t{ 1 } });
  }

} // for_each_neighbor_cell_across_face

} // namespace kalypsso

#endif // KALYPSSO_CORE_NEIGHBORCELLSACROSSFACE_H_
