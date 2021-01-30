source_filename = "test"
target datalayout = "e-m:e-p:64:64-i64:64-f80:128-n8:16:32:64-S128"

@global_var_100000f90 = local_unnamed_addr global i64 4611686018427387904
@global_var_100000f98 = local_unnamed_addr global i64 4607182418800017408
@global_var_100000fa0 = local_unnamed_addr global i64 4616189618054758400
@global_var_100000fa8 = local_unnamed_addr global i64 4617315517961601024

define i128 @_agent() local_unnamed_addr {
dec_label_pc_100000ed0:
  %storemerge.reg2mem = alloca i64, !insn.addr !0
  %0 = call i128 @__decompiler_undefined_function_1()
  %1 = call i128 @__decompiler_undefined_function_1()
  %2 = call i1 @__decompiler_undefined_function_2()
  %3 = call i1 @__decompiler_undefined_function_2()
  %4 = call i64 @__asm_movsd.1(i128 %1), !insn.addr !1
  %5 = call i64 @__asm_movsd.1(i128 %0), !insn.addr !2
  %6 = call i128 @__asm_movsd(i64 %4), !insn.addr !3
  %7 = call i128 @__asm_movsd(i64 %5), !insn.addr !4
  call void @__asm_ucomisd(i128 %7, i128 %6), !insn.addr !5
  %8 = or i1 %2, %3, !insn.addr !6
  br i1 %8, label %dec_label_pc_100000f06, label %dec_label_pc_100000ef2, !insn.addr !6

dec_label_pc_100000ef2:                           ; preds = %dec_label_pc_100000ed0
  %9 = call i128 @__asm_movsd(i64 %4), !insn.addr !7
  %10 = call i128 @__asm_mulsd(i128 %9, i64 %4), !insn.addr !8
  %11 = call i64 @__asm_movsd.1(i128 %10), !insn.addr !9
  store i64 %11, i64* %storemerge.reg2mem, !insn.addr !10
  br label %dec_label_pc_100000f18, !insn.addr !10

dec_label_pc_100000f06:                           ; preds = %dec_label_pc_100000ed0
  %12 = load i64, i64* @global_var_100000f90, align 8, !insn.addr !11
  %13 = call i128 @__asm_movsd(i64 %12), !insn.addr !11
  %14 = call i128 @__asm_mulsd(i128 %13, i64 %4), !insn.addr !12
  %15 = call i64 @__asm_movsd.1(i128 %14), !insn.addr !13
  store i64 %15, i64* %storemerge.reg2mem, !insn.addr !13
  br label %dec_label_pc_100000f18, !insn.addr !13

dec_label_pc_100000f18:                           ; preds = %dec_label_pc_100000f06, %dec_label_pc_100000ef2
  %storemerge.reload = load i64, i64* %storemerge.reg2mem
  %16 = load i64, i64* @global_var_100000f98, align 8, !insn.addr !14
  %17 = call i128 @__asm_movsd(i64 %16), !insn.addr !14
  %18 = call i128 @__asm_addsd(i128 %17, i64 %storemerge.reload), !insn.addr !15
  %19 = call i64 @__asm_movsd.1(i128 %18), !insn.addr !16
  %20 = call i128 @__asm_movsd(i64 %19), !insn.addr !17
  ret i128 %20, !insn.addr !18
}

define i64 @main(i64 %argc, i8** %argv) local_unnamed_addr {
dec_label_pc_100000f40:
  %0 = load i64, i64* @global_var_100000fa0, align 8, !insn.addr !19
  %1 = call i128 @__asm_movsd(i64 %0), !insn.addr !19
  %2 = load i64, i64* @global_var_100000fa8, align 8, !insn.addr !20
  %3 = call i128 @__asm_movsd(i64 %2), !insn.addr !20
  %4 = call i64 @__asm_movsd.1(i128 %3), !insn.addr !21
  %5 = call i64 @__asm_movsd.1(i128 %1), !insn.addr !22
  %6 = call i128 @__asm_movsd(i64 %5), !insn.addr !23
  %7 = call i128 @__asm_movsd(i64 %4), !insn.addr !24
  %8 = call i128 @_agent(), !insn.addr !25
  %9 = call i64 @__asm_movsd.1(i128 %8), !insn.addr !26
  ret i64 1, !insn.addr !27
}

declare i128 @__asm_movsd(i64) local_unnamed_addr

declare i64 @__asm_movsd.1(i128) local_unnamed_addr

declare void @__asm_ucomisd(i128, i128) local_unnamed_addr

declare i128 @__asm_mulsd(i128, i64) local_unnamed_addr

declare i128 @__asm_addsd(i128, i64) local_unnamed_addr

declare i128 @__decompiler_undefined_function_1() local_unnamed_addr

declare i1 @__decompiler_undefined_function_2() local_unnamed_addr

!0 = !{i64 4294971088}
!1 = !{i64 4294971092}
!2 = !{i64 4294971097}
!3 = !{i64 4294971102}
!4 = !{i64 4294971107}
!5 = !{i64 4294971112}
!6 = !{i64 4294971116}
!7 = !{i64 4294971122}
!8 = !{i64 4294971127}
!9 = !{i64 4294971132}
!10 = !{i64 4294971137}
!11 = !{i64 4294971142}
!12 = !{i64 4294971150}
!13 = !{i64 4294971155}
!14 = !{i64 4294971160}
!15 = !{i64 4294971168}
!16 = !{i64 4294971173}
!17 = !{i64 4294971178}
!18 = !{i64 4294971184}
!19 = !{i64 4294971208}
!20 = !{i64 4294971216}
!21 = !{i64 4294971238}
!22 = !{i64 4294971243}
!23 = !{i64 4294971248}
!24 = !{i64 4294971253}
!25 = !{i64 4294971258}
!26 = !{i64 4294971263}
!27 = !{i64 4294971278}
