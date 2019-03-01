@enum MODE PRE=1 CURR=2 POST=3
struct TestContState
    x::Float64
    term::Bool
end
const TestStateType = CMSSPState{MODE,TestContState}

@enum MODESWITCH STAY=1 SWITCH=2

const TestActionType = CMSSPAction{MODESWITCH, Float64}

const TestCMSSPType = CMSSP{MODE,TestContState,MODESWITCH,Float64}

# Conversion functions
function POMDPs.convert_s(::Type{V}, s::TestStateType, 
                          mdp::TestCMSSPType) where {V <: AbstractVector{Float64}}
    v = SVector{3,Float64}(convert(Float64, Int(s.mode)), s.continuous.x, convert(Float64, s.continuous.term))
    return v
end

function POMDPs.convert_s(::Type{TestStateType}, v::AbstractVector{Float64}, mdp::TestCMSSPType)
    mode = MODE(convert(Int64,v[1]))
    continuous = TestContState(v[2], convert(Bool,v[3]))
end