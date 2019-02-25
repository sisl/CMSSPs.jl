# This implements the functionality required by the local layer
# For now, we're doing LocalApproximationVI, so the user has to 
# provide the interpolation object and define the necessary convert functions
# struct ModalPolicy
#     in_horizon_policy::LocalApproximationValueIterationPolicy
#     out_horizon_policy::LocalApproximationValueIterationPolicy
# end

# Dict{D,ModalPolicy}

# function get_modal_policy{D,C}(cmssp, mode, interp,...)
# function compute_penalty
# function convert_s (use convert_s for underlying state)
# struct CMSSPAugmented (for horizon)
# function incorporate_closedloop_interrupt