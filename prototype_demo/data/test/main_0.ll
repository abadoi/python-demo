; ModuleID = 'main.c'
source_filename = "main.c"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

; Function Attrs: noinline nounwind optnone ssp uwtable
define double @foo(double %0, double %1) #0 !dbg !8 {
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  store double %0, double* %3, align 8
  call void @llvm.dbg.declare(metadata double* %3, metadata !12, metadata !DIExpression()), !dbg !13
  store double %1, double* %4, align 8
  call void @llvm.dbg.declare(metadata double* %4, metadata !14, metadata !DIExpression()), !dbg !15
  %5 = load double, double* %3, align 8, !dbg !16
  %6 = load double, double* %4, align 8, !dbg !18
  %7 = fcmp olt double %5, %6, !dbg !19
  br i1 %7, label %8, label %12, !dbg !20

8:                                                ; preds = %2
  %9 = load double, double* %3, align 8, !dbg !21
  %10 = load double, double* %3, align 8, !dbg !22
  %11 = fmul double %9, %10, !dbg !23
  store double %11, double* %3, align 8, !dbg !24
  br label %15, !dbg !25

12:                                               ; preds = %2
  %13 = load double, double* %3, align 8, !dbg !26
  %14 = fmul double 2.000000e+00, %13, !dbg !27
  store double %14, double* %3, align 8, !dbg !28
  br label %15

15:                                               ; preds = %12, %8
  %16 = load double, double* %3, align 8, !dbg !29
  %17 = fadd double %16, 1.000000e+00, !dbg !29
  store double %17, double* %3, align 8, !dbg !29
  %18 = load double, double* %3, align 8, !dbg !30
  ret double %18, !dbg !31
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline nounwind optnone ssp uwtable
define i32 @main(i32 %0, i8** %1) #0 !dbg !32 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i8**, align 8
  %6 = alloca double, align 8
  %7 = alloca double, align 8
  %8 = alloca double, align 8
  store i32 0, i32* %3, align 4
  store i32 %0, i32* %4, align 4
  call void @llvm.dbg.declare(metadata i32* %4, metadata !39, metadata !DIExpression()), !dbg !40
  store i8** %1, i8*** %5, align 8
  call void @llvm.dbg.declare(metadata i8*** %5, metadata !41, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.declare(metadata double* %6, metadata !43, metadata !DIExpression()), !dbg !44
  store double 5.000000e+00, double* %6, align 8, !dbg !44
  call void @llvm.dbg.declare(metadata double* %7, metadata !45, metadata !DIExpression()), !dbg !46
  store double 4.000000e+00, double* %7, align 8, !dbg !46
  call void @llvm.dbg.declare(metadata double* %8, metadata !47, metadata !DIExpression()), !dbg !48
  %9 = load double, double* %7, align 8, !dbg !49
  %10 = load double, double* %6, align 8, !dbg !50
  %11 = call double @foo(double %9, double %10), !dbg !51
  store double %11, double* %8, align 8, !dbg !48
  ret i32 1, !dbg !52
}

attributes #0 = { noinline nounwind optnone ssp uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "main.c", directory: "/Users/andreibadoi/Master_Thesis/material/scripts/pattern_matching/data/1")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 10.0.0 "}
!8 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 5, type: !9, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11, !11}
!11 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!12 = !DILocalVariable(name: "x", arg: 1, scope: !8, file: !1, line: 5, type: !11)
!13 = !DILocation(line: 5, column: 12, scope: !8)
!14 = !DILocalVariable(name: "var", arg: 2, scope: !8, file: !1, line: 5, type: !11)
!15 = !DILocation(line: 5, column: 22, scope: !8)
!16 = !DILocation(line: 8, column: 9, scope: !17)
!17 = distinct !DILexicalBlock(scope: !8, file: !1, line: 8, column: 9)
!18 = !DILocation(line: 8, column: 13, scope: !17)
!19 = !DILocation(line: 8, column: 11, scope: !17)
!20 = !DILocation(line: 8, column: 9, scope: !8)
!21 = !DILocation(line: 9, column: 13, scope: !17)
!22 = !DILocation(line: 9, column: 17, scope: !17)
!23 = !DILocation(line: 9, column: 15, scope: !17)
!24 = !DILocation(line: 9, column: 11, scope: !17)
!25 = !DILocation(line: 9, column: 9, scope: !17)
!26 = !DILocation(line: 11, column: 19, scope: !17)
!27 = !DILocation(line: 11, column: 17, scope: !17)
!28 = !DILocation(line: 11, column: 11, scope: !17)
!29 = !DILocation(line: 12, column: 7, scope: !8)
!30 = !DILocation(line: 14, column: 12, scope: !8)
!31 = !DILocation(line: 14, column: 5, scope: !8)
!32 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 17, type: !33, scopeLine: 18, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!33 = !DISubroutineType(types: !34)
!34 = !{!35, !35, !36}
!35 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !37, size: 64)
!37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !38, size: 64)
!38 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!39 = !DILocalVariable(name: "argc", arg: 1, scope: !32, file: !1, line: 17, type: !35)
!40 = !DILocation(line: 17, column: 14, scope: !32)
!41 = !DILocalVariable(name: "argv", arg: 2, scope: !32, file: !1, line: 17, type: !36)
!42 = !DILocation(line: 17, column: 27, scope: !32)
!43 = !DILocalVariable(name: "thres", scope: !32, file: !1, line: 19, type: !11)
!44 = !DILocation(line: 19, column: 12, scope: !32)
!45 = !DILocalVariable(name: "x", scope: !32, file: !1, line: 20, type: !11)
!46 = !DILocation(line: 20, column: 12, scope: !32)
!47 = !DILocalVariable(name: "result", scope: !32, file: !1, line: 21, type: !11)
!48 = !DILocation(line: 21, column: 12, scope: !32)
!49 = !DILocation(line: 21, column: 25, scope: !32)
!50 = !DILocation(line: 21, column: 28, scope: !32)
!51 = !DILocation(line: 21, column: 21, scope: !32)
!52 = !DILocation(line: 23, column: 5, scope: !32)
