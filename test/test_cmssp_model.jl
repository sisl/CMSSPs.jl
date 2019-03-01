let
    actions = Vector{TestActionType}(undef,0)
    push!(actions,TestActionType(STAY,1))
    push!(actions, TestActionType(0.5,2))
    cmssp = TestCMSSPType(actions, [PRE,CURR,POST])

    tsv = TestStateType(CURR, TestContState(0.4,false))
    @test n_actions(cmssp) == 2
    @test discount(cmssp) == 1.0
    @test convert_s(Vector{Float64},tsv,cmssp) == [2.0,0.4,0.0]
    @test get_modeswitch_actions(cmssp.actions) == [STAY]
    @test get_control_actions(cmssp.actions) == [0.5]
    olv = OpenLoopVertex{MODE,TestContState,MODESWITCH}(tsv,STAY)
end


# TODO Tests
# More complicated ordering of actions (in case alternating AD and AC)