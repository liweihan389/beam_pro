from beamic_sdk import Plugin


class AnalysisPlugin(Plugin):
    def on_activate(self):
        self.last_id = -1
        self.host.panel("基础光斑分析", "等待未饱和的有效边界拟合。")

    def on_frame(self, frame, fit):
        if frame.frame_id == self.last_id:
            return
        self.last_id = frame.frame_id
        if not frame.exposure_verified or not frame.saturation_verified or not frame.metadata.get("quantitative_ready"):
            self.host.panel("基础光斑分析", "设备量程或逐帧曝光尚未核对，当前数据仅供预览。")
        elif fit and fit.valid:
            ellipticity = fit.radius_minor / fit.radius_major
            mode = "边界拟合中心" if fit.center_mode == "boundary" else "面积中心"
            self.host.panel("基础光斑分析", f"{fit.boundary_fraction:.1%} 边界长半轴：{fit.radius_major:.2f} px\n"
                            f"短半轴：{fit.radius_minor:.2f} px\n椭圆率：{ellipticity:.3f}\n"
                            f"中心定义：{mode}\n边界几何相对残差：{fit.residual_fraction:.3%}\n"
                            f"{fit.reason}\n"
                            "此示例不测量 M² 或实际耦合效率。")
        else:
            self.host.panel("基础光斑分析", fit.reason if fit else "等待拟合")

