let
    actions = Vector{TestActionType}(undef,0)
    push!(actions,TestActionType(STAY,1))
    push!(actions, TestActionType(0.5,2))
    start_state = TestStateType(PRE, TestContState(0.35,false))
    goal_state = TestStateType(POST, TestContState(0.5,true))
    cmssp = TestCMSSPType(actions,start_state,POST)

    tsv = TestStateType(CURR, TestContState(0.4,false))
    @test n_actions(cmssp) == 2
    @test discount(cmssp) == 1.0
    @test convert_s(Vector{Float64},tsv,cmssp) == [2.0,0.4,0.0]
end