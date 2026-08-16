`timescale 1ns/1ps
module tb_frozen_ternary_dot;
    logic [31:0] activations;
    logic signed [5:0] result;
    integer i;
    integer expected;
    reg [1:0] activation_code;
    reg [1:0] weight_code;
    frozen_ternary_dot_16 dut(.activations(activations), .result(result));
    initial begin
        activations = '0;
        #1;
        if (result !== 0) $fatal(1, "zero vector mismatch");
        for (i = 0; i < 16; i = i + 1)
            activations[2*i +: 2] = (i % 3 == 0) ? 2'b00 :
                                             (i % 3 == 1) ? 2'b01 : 2'b10;
        expected = 0;
        #1;
        for (i = 0; i < 16; i = i + 1) begin
            weight_code = dut.WEIGHTS[2*i +: 2];
            activation_code = activations[2*i +: 2];
            if ((weight_code == 2'b01 && activation_code == 2'b01) ||
                (weight_code == 2'b10 && activation_code == 2'b10)) expected = expected + 1;
            if ((weight_code == 2'b01 && activation_code == 2'b10) ||
                (weight_code == 2'b10 && activation_code == 2'b01)) expected = expected - 1;
        end
        if (result !== expected) $fatal(1, "pattern mismatch got=%0d expected=%0d", result, expected);
        $display("PASS frozen ternary dot: result=%0d", result);
        $finish;
    end
endmodule
